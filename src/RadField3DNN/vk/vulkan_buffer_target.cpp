#include <radfield3d-nn/vk/vulkan_buffer_target.h>

#include <radfield3d-nn/field_predictors.h>
#include <radfield3d-nn/model_binding.h>
#include <radfield3d-nn/vk/cuda_vulkan_export.h>

#include <cuda_runtime_api.h>

#include <cstdio>

namespace rfnn {
namespace vk {

// The registration: the buffer's exported VkDeviceMemory is imported as CUDA external memory and
// mapped to a plain linear device pointer. That pointer aliases the Vulkan allocation, so binding
// it as a model output makes the inference kernels write the Vulkan buffer itself.
struct VulkanBufferTarget::Impl {
    cudaExternalMemory_t ext_mem = nullptr;
    void*                dptr    = nullptr;
    std::size_t          bytes   = 0;
    int                  device  = -1;

    ~Impl() {
        if (device >= 0) cudaSetDevice(device);
        if (dptr)    cudaFree(dptr);                       // releases the mapping, not the memory
        if (ext_mem) cudaDestroyExternalMemory(ext_mem);   // the Vulkan side still owns the allocation
    }
};

VulkanBufferTarget::VulkanBufferTarget() = default;
VulkanBufferTarget::~VulkanBufferTarget() = default;
VulkanBufferTarget::VulkanBufferTarget(VulkanBufferTarget&&) noexcept = default;
VulkanBufferTarget& VulkanBufferTarget::operator=(VulkanBufferTarget&&) noexcept = default;

// Shared registration: import the allocation behind the buffer and map the buffer's span to a
// linear device pointer. Buffers are NOT dedicated allocations — no cudaExternalMemoryDedicated,
// unlike the image path.
VulkanBufferTarget VulkanBufferTarget::register_imported(const uint8_t device_uuid[16],
                                                         const void* handle_desc,
                                                         std::size_t mem_offset, std::size_t bytes) {
    const auto& mem_desc = *static_cast<const cudaExternalMemoryHandleDesc*>(handle_desc);
    VulkanBufferTarget t;

    const int device = rfnn::cuda_vk::cuda_device_for_uuid(device_uuid);
    if (device < 0) {
        std::fprintf(stderr, "[rfnn::vk] no CUDA device matches the Vulkan device UUID\n");
        return t;
    }
    if (cudaSetDevice(device) != cudaSuccess) return t;

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->bytes  = bytes;

    if (cudaImportExternalMemory(&impl->ext_mem, &mem_desc) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaImportExternalMemory failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return t;
    }

    cudaExternalMemoryBufferDesc buf_desc{};
    buf_desc.offset = mem_offset;
    buf_desc.size   = bytes;
    if (cudaExternalMemoryGetMappedBuffer(&impl->dptr, impl->ext_mem, &buf_desc) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaExternalMemoryGetMappedBuffer failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return t;
    }

    t.impl_ = std::move(impl);
    return t;
}

VulkanBufferTarget VulkanBufferTarget::register_buffer(const uint8_t device_uuid[16],
                                                       int mem_fd, std::size_t mem_size,
                                                       std::size_t mem_offset, std::size_t bytes) {
    // The FD is consumed by CUDA even on failure (POSIX external-memory semantics).
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = mem_fd;
    mem_desc.size      = mem_size;
    return register_imported(device_uuid, &mem_desc, mem_offset, bytes);
}

VulkanBufferTarget VulkanBufferTarget::register_buffer_win32(const uint8_t device_uuid[16],
                                                             void* mem_handle, std::size_t mem_size,
                                                             std::size_t mem_offset, std::size_t bytes) {
    // Win32 handles are SHARED, not consumed — the caller keeps ownership (see the header).
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    mem_desc.handle.win32.handle = mem_handle;
    mem_desc.size                = mem_size;
    return register_imported(device_uuid, &mem_desc, mem_offset, bytes);
}

void VulkanBufferTarget::bind_output(radfield3dnn::VolumeFieldPredictor& predictor,
                                     radfield3dnn::ModelOutput flag, bool convert_buffer) const {
    if (!impl_)
        throw radfield3dnn::BindingError(
            "VulkanBufferTarget::bind_output: the target is empty (register_buffer failed) — "
            "nothing can be bound. Fall back to a host/CUDA buffer of your own.");
    // From here the standard binding contract applies: capacity is checked in elements against the
    // predictor's grid, and a device/precision mismatch with the session is refused with the full
    // diagnostic unless convert_buffer opts in.
    radfield3dnn::MemoryRef ref;
    ref.data      = impl_->dptr;
    ref.bytes     = impl_->bytes;
    ref.device    = radfield3dnn::DeviceKind::Cuda;
    ref.dtype     = radfield3dnn::ElementType::F32;
    ref.device_id = impl_->device;
    predictor.bind_output_layer(flag, ref, convert_buffer);
}

std::size_t VulkanBufferTarget::capacity_bytes() const { return impl_ ? impl_->bytes : 0; }
int VulkanBufferTarget::device_index() const { return impl_ ? impl_->device : -1; }
VulkanBufferTarget::operator bool() const { return impl_ != nullptr; }

}  // namespace vk
}  // namespace rfnn
