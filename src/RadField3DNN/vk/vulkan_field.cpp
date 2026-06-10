#include "radfield3d-nn/vk/vulkan_field.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <RadFiled3D/Voxel.hpp>

// SPIR-V compute shaders, embedded as C arrays by the build (glslangValidator --vn <symbol>).
#include "generated/airkerma_combine_spv.h"    // airkerma_combine_spv[]
#include "generated/voxel_visibility_spv.h"    // voxel_visibility_spv[]

namespace rfnn::vk {
using namespace radfield3dnn;
namespace {

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string("vulkan_field: ") + what +
                                                  " failed (VkResult " + std::to_string(r) + ")");
}

uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_bits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((type_bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("vulkan_field: no suitable memory type");
}

// A small owned (buffer, memory) pair the helpers manage internally.
struct Alloc {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

Alloc create_buffer(const VulkanContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags props) {
    Alloc a; a.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(ctx.device, &bi, nullptr, &a.buffer), "vkCreateBuffer");

    VkMemoryRequirements req; vkGetBufferMemoryRequirements(ctx.device, a.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory_type(ctx.physical_device, req.memoryTypeBits, props);
    check(vkAllocateMemory(ctx.device, &ai, nullptr, &a.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(ctx.device, a.buffer, a.memory, 0), "vkBindBufferMemory");
    return a;
}

void free_alloc(const VulkanContext& ctx, Alloc& a) {
    if (a.buffer) vkDestroyBuffer(ctx.device, a.buffer, nullptr);
    if (a.memory) vkFreeMemory(ctx.device, a.memory, nullptr);
    a.buffer = VK_NULL_HANDLE; a.memory = VK_NULL_HANDLE;
}

// One-shot command buffer: allocate, record via `record`, submit, wait, free.
template <class Fn>
void submit_once(const VulkanContext& ctx, Fn&& record) {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.queueFamilyIndex = ctx.queue_family;
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VkCommandPool pool; check(vkCreateCommandPool(ctx.device, &pci, nullptr, &pool), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(ctx.device, &cbi, &cmd), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
    record(cmd);
    check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

    VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence; check(vkCreateFence(ctx.device, &fi, nullptr, &fence), "vkCreateFence");
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    check(vkQueueSubmit(ctx.queue, 1, &si, fence), "vkQueueSubmit");
    check(vkWaitForFences(ctx.device, 1, &fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");

    vkDestroyFence(ctx.device, fence, nullptr);
    vkDestroyCommandPool(ctx.device, pool, nullptr);  // frees the command buffer too
}

// Upload `bytes` host data into device-local `dst` through a temporary staging buffer.
void upload(const VulkanContext& ctx, VkBuffer dst, const void* src, VkDeviceSize bytes) {
    Alloc staging = create_buffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    check(vkMapMemory(ctx.device, staging.memory, 0, bytes, 0, &mapped), "vkMapMemory");
    std::memcpy(mapped, src, static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.device, staging.memory);

    submit_once(ctx, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{}; region.size = bytes;
        vkCmdCopyBuffer(cmd, staging.buffer, dst, 1, &region);
    });
    free_alloc(ctx, staging);
}

// Resolve the destination buffer for an upload helper: either the caller's `target` (we do
// not own it) or a freshly allocated device-local storage buffer (we do).
VulkanBuffer resolve_target(const VulkanContext& ctx, VkBuffer target, VkDeviceSize bytes,
                            uint32_t element_count) {
    VulkanBuffer out; out.size_bytes = bytes; out.element_count = element_count;
    if (target != VK_NULL_HANDLE) {
        out.buffer = target; out.owns_memory = false;
    } else {
        Alloc a = create_buffer(ctx, bytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        out.buffer = a.buffer; out.memory = a.memory; out.owns_memory = true;
    }
    return out;
}

// Fetch a layer's contiguous float base pointer from the prediction channel.
const float* layer_floats(const DeviceCartesianRadiationField& field, const std::string& channel,
                          const std::string& layer) {
    auto buf = field.get_channel(channel);
    return buf->get_layer<float>(layer);
}

size_t spectrum_bins(const DeviceCartesianRadiationField& field, const std::string& channel) {
    auto buf = field.get_channel(channel);
    auto& v = buf->get_voxel_flat<RadFiled3D::HistogramVoxel<float>>(kSpectrumLayer, 0);
    return v.get_bins();
}

}  // namespace

void destroy(const VulkanContext& ctx, VulkanBuffer& buf) {
    if (buf.owns_memory) {
        if (buf.buffer) vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
        if (buf.memory) vkFreeMemory(ctx.device, buf.memory, nullptr);
    }
    buf = VulkanBuffer{};
}

std::vector<float> download_floats(const VulkanContext& ctx, VkBuffer src, uint32_t count) {
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float);
    Alloc staging = create_buffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    submit_once(ctx, [&](VkCommandBuffer cmd) {
        VkBufferCopy region{}; region.size = bytes;
        vkCmdCopyBuffer(cmd, src, staging.buffer, 1, &region);
    });
    std::vector<float> out(count);
    void* mapped = nullptr;
    check(vkMapMemory(ctx.device, staging.memory, 0, bytes, 0, &mapped), "vkMapMemory");
    std::memcpy(out.data(), mapped, static_cast<size_t>(bytes));
    vkUnmapMemory(ctx.device, staging.memory);
    free_alloc(ctx, staging);
    return out;
}

VulkanBuffer write_flux_to_vulkan(const VulkanContext& ctx,
                                  const DeviceCartesianRadiationField& field,
                                  const std::string& channel, VkBuffer target) {
    const uint32_t n = static_cast<uint32_t>(field.voxel_count());
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * sizeof(float);
    VulkanBuffer out = resolve_target(ctx, target, bytes, n);
    upload(ctx, out.buffer, layer_floats(field, channel, kFluxLayer), bytes);
    return out;
}

VulkanBuffer write_spectrum_to_vulkan(const VulkanContext& ctx,
                                      const DeviceCartesianRadiationField& field,
                                      const std::string& channel, VkBuffer target) {
    const uint32_t n = static_cast<uint32_t>(field.voxel_count());
    const uint32_t bins = static_cast<uint32_t>(spectrum_bins(field, channel));
    const uint32_t elems = n * bins;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(elems) * sizeof(float);
    VulkanBuffer out = resolve_target(ctx, target, bytes, elems);
    upload(ctx, out.buffer, layer_floats(field, channel, kSpectrumLayer), bytes);
    return out;
}

VulkanBuffer write_airkerma_to_vulkan(const VulkanContext& ctx,
                                      const DeviceCartesianRadiationField& field,
                                      const std::vector<float>& kerma_coeff,
                                      const std::string& channel, VkBuffer target) {
    const uint32_t n = static_cast<uint32_t>(field.voxel_count());
    const uint32_t bins = static_cast<uint32_t>(spectrum_bins(field, channel));
    if (kerma_coeff.size() != bins)
        throw std::runtime_error("write_airkerma_to_vulkan: kerma_coeff length (" +
                                 std::to_string(kerma_coeff.size()) + ") != spectrum bins (" +
                                 std::to_string(bins) + ")");

    // Inputs as device-local SSBOs (uploaded once, freed at the end).
    Alloc flux = create_buffer(ctx, VkDeviceSize(n) * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Alloc spec = create_buffer(ctx, VkDeviceSize(n) * bins * sizeof(float),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    Alloc coeff = create_buffer(ctx, VkDeviceSize(bins) * sizeof(float),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    upload(ctx, flux.buffer, layer_floats(field, channel, kFluxLayer), flux.size);
    upload(ctx, spec.buffer, layer_floats(field, channel, kSpectrumLayer), spec.size);
    upload(ctx, coeff.buffer, kerma_coeff.data(), coeff.size);

    VulkanBuffer out = resolve_target(ctx, target, VkDeviceSize(n) * sizeof(float), n);

    // Descriptor set layout: 4 storage buffers (flux, spectrum, coeff, airkerma).
    VkDescriptorSetLayoutBinding binds[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsl.bindingCount = 4; dsl.pBindings = binds;
    VkDescriptorSetLayout set_layout; check(vkCreateDescriptorSetLayout(ctx.device, &dsl, nullptr, &set_layout), "DescriptorSetLayout");

    VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0; pcr.size = 2 * sizeof(uint32_t);
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipe_layout; check(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipe_layout), "PipelineLayout");

    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sizeof(airkerma_combine_spv); smi.pCode = airkerma_combine_spv;
    VkShaderModule module; check(vkCreateShaderModule(ctx.device, &smi, nullptr, &module), "ShaderModule");

    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = module; cpi.stage.pName = "main";
    cpi.layout = pipe_layout;
    VkPipeline pipeline; check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline), "ComputePipeline");

    // Descriptor pool + set.
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
    VkDescriptorPool dpool; check(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool), "DescriptorPool");
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &set_layout;
    VkDescriptorSet dset; check(vkAllocateDescriptorSets(ctx.device, &dsai, &dset), "AllocateDescriptorSets");

    VkBuffer bufs[4] = {flux.buffer, spec.buffer, coeff.buffer, out.buffer};
    VkDescriptorBufferInfo dbi[4]{}; VkWriteDescriptorSet wds[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        dbi[i].buffer = bufs[i]; dbi[i].offset = 0; dbi[i].range = VK_WHOLE_SIZE;
        wds[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wds[i].dstSet = dset;
        wds[i].dstBinding = i; wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(ctx.device, 4, wds, 0, nullptr);

    const uint32_t pc[2] = {n, bins};
    submit_once(ctx, [&](VkCommandBuffer cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &dset, 0, nullptr);
        vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(cmd, (n + 63u) / 64u, 1, 1);
    });

    vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
    vkDestroyPipeline(ctx.device, pipeline, nullptr);
    vkDestroyShaderModule(ctx.device, module, nullptr);
    vkDestroyPipelineLayout(ctx.device, pipe_layout, nullptr);
    vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
    free_alloc(ctx, flux); free_alloc(ctx, spec); free_alloc(ctx, coeff);
    return out;
}

// Persistent state for VoxelVisibilityCuller: the compute pipeline + reusable, persistently-mapped
// HOST-VISIBLE buffers (so the survivors can be read / bound to ONNX with no download).
struct VoxelVisibilityCuller::Impl {
    struct PC { uint32_t D, H, W, pad; float proj[16]; };  // push constant (80 bytes)

    VulkanContext         ctx;
    VkDescriptorSetLayout set_layout  = VK_NULL_HANDLE;
    VkPipelineLayout      pipe_layout = VK_NULL_HANDLE;
    VkShaderModule        module      = VK_NULL_HANDLE;
    VkPipeline            pipeline    = VK_NULL_HANDLE;
    VkDescriptorPool      dpool       = VK_NULL_HANDLE;
    VkDescriptorSet       dset        = VK_NULL_HANDLE;
    Alloc      counter, positions, indices;     // all host-visible coherent, persistently mapped
    uint32_t*  counter_map   = nullptr;
    float*     positions_map = nullptr;          // [capacity*3] x,y,z
    uint32_t*  indices_map   = nullptr;          // [capacity] flat voxel index
    uint32_t   capacity = 0;                     // survivor capacity in voxels

    Alloc host_buffer(VkDeviceSize size) {
        return create_buffer(ctx, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    void* map(const Alloc& a) { void* p = nullptr; check(vkMapMemory(ctx.device, a.memory, 0, a.size, 0, &p), "vkMapMemory"); return p; }
    
    void bind(uint32_t binding, VkBuffer buf) {
        VkDescriptorBufferInfo dbi{};
        dbi.buffer = buf;
        dbi.offset = 0;
        dbi.range = VK_WHOLE_SIZE;

        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = dset;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(ctx.device, 1, &w, 0, nullptr);
    }

    explicit Impl(const VulkanContext& c) : ctx(c) {
        VkDescriptorSetLayoutBinding binds[3]{};
        for (uint32_t i = 0; i < 3; ++i) {
            binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dsl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dsl.bindingCount = 3; dsl.pBindings = binds;
        check(vkCreateDescriptorSetLayout(ctx.device, &dsl, nullptr, &set_layout), "DescriptorSetLayout");

        VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pcr.size = sizeof(PC);
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &set_layout; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        check(vkCreatePipelineLayout(ctx.device, &pli, nullptr, &pipe_layout), "PipelineLayout");

        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = sizeof(voxel_visibility_spv); smi.pCode = voxel_visibility_spv;
        check(vkCreateShaderModule(ctx.device, &smi, nullptr, &module), "ShaderModule");

        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = module; cpi.stage.pName = "main";
        cpi.layout = pipe_layout;
        check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipeline), "ComputePipeline");

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &ps;
        check(vkCreateDescriptorPool(ctx.device, &dpi, nullptr, &dpool), "DescriptorPool");
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &set_layout;
        check(vkAllocateDescriptorSets(ctx.device, &dsai, &dset), "AllocateDescriptorSets");

        counter = host_buffer(sizeof(uint32_t));
        counter_map = static_cast<uint32_t*>(map(counter));
        bind(0, counter.buffer);
    }

    ~Impl() {
        if (counter.memory)   vkUnmapMemory(ctx.device, counter.memory);
        if (positions.memory) vkUnmapMemory(ctx.device, positions.memory);
        if (indices.memory)   vkUnmapMemory(ctx.device, indices.memory);
        if (dpool)       vkDestroyDescriptorPool(ctx.device, dpool, nullptr);
        if (pipeline)    vkDestroyPipeline(ctx.device, pipeline, nullptr);
        if (module)      vkDestroyShaderModule(ctx.device, module, nullptr);
        if (pipe_layout) vkDestroyPipelineLayout(ctx.device, pipe_layout, nullptr);
        if (set_layout)  vkDestroyDescriptorSetLayout(ctx.device, set_layout, nullptr);
        free_alloc(ctx, counter); free_alloc(ctx, positions); free_alloc(ctx, indices);
    }

    void ensure_capacity(uint32_t N) {
        if (positions.buffer && N <= capacity) return;
        if (positions.memory) { vkUnmapMemory(ctx.device, positions.memory); free_alloc(ctx, positions); }
        if (indices.memory)   { vkUnmapMemory(ctx.device, indices.memory);   free_alloc(ctx, indices); }
        positions = host_buffer(VkDeviceSize(N) * 3 * sizeof(float));
        indices   = host_buffer(VkDeviceSize(N) * sizeof(uint32_t));
        positions_map = static_cast<float*>(map(positions));
        indices_map   = static_cast<uint32_t*>(map(indices));
        capacity = N;

        // bind output buffer to layout defined in shader
        bind(1, positions.buffer);
        bind(2, indices.buffer);
    }

    // Dispatch the cull and return the survivor count; positions/indices are in the mapped buffers.
    uint32_t run(std::array<int, 3> dims, const std::array<float, 16>& projection) {
        const uint32_t D = static_cast<uint32_t>(std::max(0, dims[0]));
        const uint32_t H = static_cast<uint32_t>(std::max(0, dims[1]));
        const uint32_t W = static_cast<uint32_t>(std::max(0, dims[2]));
        const uint64_t N64 = static_cast<uint64_t>(D) * H * W;
        if (N64 == 0) return 0;
        const uint32_t N = static_cast<uint32_t>(N64);
        ensure_capacity(N);
        *counter_map = 0u;   // host-coherent: visible to the GPU without an explicit flush

        PC pc{D, H, W, 0u, {}};
        std::memcpy(pc.proj, projection.data(), sizeof(pc.proj));
        submit_once(ctx, [&](VkCommandBuffer cmd) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &dset, 0, nullptr);
            vkCmdPushConstants(cmd, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, (N + 63u) / 64u, 1, 1);
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        });
        return std::min(*counter_map, N);
    }
};

VoxelVisibilityCuller::VoxelVisibilityCuller(const VulkanContext& ctx)
    : impl_(std::make_unique<Impl>(ctx)) {}
VoxelVisibilityCuller::~VoxelVisibilityCuller() = default;

VisibleVoxels VoxelVisibilityCuller::cull(std::array<int, 3> dims, const std::array<float, 16>& projection) {
    Impl& im = *impl_;
    VisibleVoxels v;
    v.count     = im.run(dims, projection);   // survivors are now in the mapped host buffers
    v.positions = im.positions_map;
    v.indices   = im.indices_map;
    return v;
}

std::vector<std::array<float, 3>> compute_visible_voxels(const VulkanContext& ctx,
                                                         std::array<int, 3> dims,
                                                         const std::array<float, 16>& projection) {
    // One-shot wrapper. For a per-frame loop build a VoxelVisibilityCuller once and reuse it.
    VoxelVisibilityCuller culler(ctx);
    const VisibleVoxels v = culler.cull(dims, projection);
    std::vector<std::array<float, 3>> pts(v.count);
    for (uint32_t i = 0; i < v.count; ++i)
        pts[i] = {v.positions[3*i], v.positions[3*i + 1], v.positions[3*i + 2]};
    return pts;
}

}  // namespace rfnn::vk
