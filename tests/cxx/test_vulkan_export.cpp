// Headless Vulkan export test: fill a small DeviceCartesianRadiationField with a known
// flux/spectrum pattern, push each export helper through a self-created (no-surface) Vulkan
// compute context, read the device buffers back, and check them against host references —
// including the air-kerma combine compute shader and the write-into-an-existing-buffer path.
// Skips cleanly when no Vulkan device is available.

#include <gtest/gtest.h>

#include <vulkan/vulkan.h>

#include <cstring>
#include <vector>

#include "radfield3d-nn/device_export.h"

using namespace radfield3dnn;
using namespace rfnn::vk;

namespace {

// Minimal headless compute context. UE5 would supply its own; for the test we make one.
struct Headless {
    VkInstance instance = VK_NULL_HANDLE;
    VkDevice   device   = VK_NULL_HANDLE;
    VulkanContext ctx;
    bool ok = false;

    bool init() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) return false;

        uint32_t n = 0; vkEnumeratePhysicalDevices(instance, &n, nullptr);
        if (n == 0) return false;
        std::vector<VkPhysicalDevice> phys(n);
        vkEnumeratePhysicalDevices(instance, &n, phys.data());

        for (auto pd : phys) {
            uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
            std::vector<VkQueueFamilyProperties> qf(qn);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
            for (uint32_t i = 0; i < qn; ++i) {
                if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    float prio = 1.f;
                    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
                    qci.queueFamilyIndex = i; qci.queueCount = 1; qci.pQueuePriorities = &prio;
                    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
                    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
                    if (vkCreateDevice(pd, &dci, nullptr, &device) != VK_SUCCESS) continue;
                    ctx.instance = instance; ctx.physical_device = pd; ctx.device = device;
                    ctx.queue_family = i;
                    vkGetDeviceQueue(device, i, 0, &ctx.queue);
                    ok = true;
                    return true;
                }
            }
        }
        return false;
    }

    ~Headless() {
        if (device)   vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// Build an 8^3 field and fill flux + spectrum with a deterministic pattern.
std::shared_ptr<DeviceCartesianRadiationField> make_field(uint32_t& n_out, uint32_t bins) {
    auto field = DeviceCartesianRadiationField::from_dims({8, 8, 8});
    const size_t n = field->voxel_count();
    n_out = static_cast<uint32_t>(n);
    auto ch = field->add_channel(kPredictionChannel);
    ch->add_layer<float>(kFluxLayer, 0.f, "flux");
    ch->add_custom_layer<RadFiled3D::HistogramVoxel<float>, float>(
        kSpectrumLayer, RadFiled3D::HistogramVoxel<float>(bins, 1.f, nullptr), 0.f, "spectrum");

    float* flux = ch->get_layer<float>(kFluxLayer);
    float* spec = ch->get_layer<float>(kSpectrumLayer);
    for (size_t v = 0; v < n; ++v) {
        flux[v] = 0.001f * static_cast<float>(v);
        for (uint32_t b = 0; b < bins; ++b)
            spec[v * bins + b] = static_cast<float>(b + 1) + 0.1f * static_cast<float>(v % 5);
    }
    return field;
}

}  // namespace

TEST(VulkanExport, FluxSpectrumAirkermaRoundTrip) {
    Headless hl;
    if (!hl.init()) GTEST_SKIP() << "no Vulkan compute device available";

    const uint32_t bins = 4;
    uint32_t n = 0;
    auto field = make_field(n, bins);
    const float* flux = field->get_channel(kPredictionChannel)->get_layer<float>(kFluxLayer);
    const float* spec = field->get_channel(kPredictionChannel)->get_layer<float>(kSpectrumLayer);

    // Flux upload (allocate path).
    {
        VulkanBuffer b = write_flux_to_vulkan(hl.ctx, *field);
        ASSERT_EQ(b.element_count, n);
        auto got = download_floats(hl.ctx, b.buffer, n);
        for (uint32_t v = 0; v < n; ++v) EXPECT_FLOAT_EQ(got[v], flux[v]);
        destroy(hl.ctx, b);
    }
    // Spectrum upload.
    {
        VulkanBuffer b = write_spectrum_to_vulkan(hl.ctx, *field);
        ASSERT_EQ(b.element_count, n * bins);
        auto got = download_floats(hl.ctx, b.buffer, n * bins);
        for (uint32_t i = 0; i < n * bins; ++i) EXPECT_FLOAT_EQ(got[i], spec[i]);
        destroy(hl.ctx, b);
    }
    // Air-kerma combine (compute shader) vs CPU reference.
    {
        std::vector<float> coeff = {0.5f, 1.0f, 2.0f, 0.25f};  // length == bins
        VulkanBuffer b = write_airkerma_to_vulkan(hl.ctx, *field, coeff);
        ASSERT_EQ(b.element_count, n);
        auto got = download_floats(hl.ctx, b.buffer, n);
        for (uint32_t v = 0; v < n; ++v) {
            float acc = 0.f;
            for (uint32_t bb = 0; bb < bins; ++bb) acc += spec[v * bins + bb] * coeff[bb];
            EXPECT_NEAR(got[v], flux[v] * acc, 1e-3f);
        }
        destroy(hl.ctx, b);
    }
}

TEST(VulkanExport, WritesIntoExistingTargetBuffer) {
    Headless hl;
    if (!hl.init()) GTEST_SKIP() << "no Vulkan compute device available";

    uint32_t n = 0;
    auto field = make_field(n, 4);
    const float* flux = field->get_channel(kPredictionChannel)->get_layer<float>(kFluxLayer);

    // Caller-owned target: device-local STORAGE|TRANSFER_DST|TRANSFER_SRC, N floats.
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = VkDeviceSize(n) * sizeof(float);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer target = VK_NULL_HANDLE;
    ASSERT_EQ(vkCreateBuffer(hl.device, &bi, nullptr, &target), VK_SUCCESS);
    VkMemoryRequirements req; vkGetBufferMemoryRequirements(hl.device, target, &req);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(hl.ctx.physical_device, &mp);
    uint32_t mt = 0; for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mt = i; break; }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size; ai.memoryTypeIndex = mt;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    ASSERT_EQ(vkAllocateMemory(hl.device, &ai, nullptr, &mem), VK_SUCCESS);
    ASSERT_EQ(vkBindBufferMemory(hl.device, target, mem, 0), VK_SUCCESS);

    VulkanBuffer b = write_flux_to_vulkan(hl.ctx, *field, kPredictionChannel, target);
    EXPECT_FALSE(b.owns_memory);
    EXPECT_EQ(b.buffer, target);
    auto got = download_floats(hl.ctx, target, n);
    for (uint32_t v = 0; v < n; ++v) EXPECT_FLOAT_EQ(got[v], flux[v]);

    vkDestroyBuffer(hl.device, target, nullptr);
    vkFreeMemory(hl.device, mem, nullptr);
}
