#include <radfield3d-nn/vk/vulkan_image_target.h>

#include <radfield3d-nn/field_predictors.h>
#include <radfield3d-nn/model_binding.h>
#include <radfield3d-nn/vk/cuda_vulkan_export.h>   // rfnn::cuda_vk::cuda_device_for_uuid (UUID → ordinal)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace rfnn {
namespace vk {

// The registration: the image's exported VkDeviceMemory is imported as CUDA external memory and mapped
// as a single-level cudaArray (dedicated allocation, unlike the buffer path). A persistent linear CUDA
// staging buffer of N elements is the model's actual output; the post-run sink relayouts it into the
// array (cudaMemcpy3D linear→array, which handles the optimal-tiling relayout) after every run.
struct VulkanImageTarget::Impl {
    int                  device     = -1;
    cudaExternalMemory_t ext_mem    = nullptr;
    cudaMipmappedArray_t mipmap     = nullptr;
    cudaArray_t          level0     = nullptr;   // mip 0 — the cudaMemcpy3D destination
    void*                staging    = nullptr;   // persistent linear device buffer (N elements)
    int                  nx = 0, ny = 0, nz = 0;
    bool                 fp16       = false;
    std::size_t          elem_bytes = 4;
    std::size_t          n          = 0;         // nx*ny*nz

    ~Impl() {
        if (device >= 0) cudaSetDevice(device);
        if (staging) cudaFree(staging);
        if (mipmap)  cudaFreeMipmappedArray(mipmap);       // frees the mapped array view
        if (ext_mem) cudaDestroyExternalMemory(ext_mem);   // the Vulkan side still owns the allocation
    }

    // Relayout staging (linear) → the imported image array, synchronously. Called by the post-run sink.
    bool relayout() {
        if (!level0 || !staging) return false;
        if (cudaSetDevice(device) != cudaSuccess) return false;
        cudaMemcpy3DParms p{};
        p.srcPtr   = make_cudaPitchedPtr(staging,
                                         (std::size_t)nx * elem_bytes,   // row pitch (bytes)
                                         (std::size_t)nx,                // width  (elements)
                                         (std::size_t)ny);               // height (rows)
        p.dstArray = level0;
        p.extent   = make_cudaExtent((std::size_t)nx, (std::size_t)ny, (std::size_t)nz);  // width in ELEMENTS
        p.kind     = cudaMemcpyDeviceToDevice;
        if (cudaMemcpy3D(&p) != cudaSuccess) {
            std::fprintf(stderr, "[rfnn::vk] image relayout cudaMemcpy3D failed: %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            return false;
        }
        // Synchronous: when predict returns, the image has been written (no cross-API semaphore in v1).
        cudaDeviceSynchronize();
        return true;
    }
};

VulkanImageTarget::VulkanImageTarget() = default;
VulkanImageTarget::~VulkanImageTarget() = default;
VulkanImageTarget::VulkanImageTarget(VulkanImageTarget&&) noexcept = default;
VulkanImageTarget& VulkanImageTarget::operator=(VulkanImageTarget&&) noexcept = default;

VulkanImageTarget VulkanImageTarget::register_imported(const uint8_t device_uuid[16],
                                                       const void* handle_desc, std::size_t mem_offset,
                                                       int dim_x, int dim_y, int dim_z, bool fp16) {
    const auto& mem_desc = *static_cast<const cudaExternalMemoryHandleDesc*>(handle_desc);
    VulkanImageTarget t;

    const int device = rfnn::cuda_vk::cuda_device_for_uuid(device_uuid);
    if (device < 0) {
        std::fprintf(stderr, "[rfnn::vk] no CUDA device matches the Vulkan device UUID\n");
        return t;
    }
    if (cudaSetDevice(device) != cudaSuccess) return t;

    auto impl = std::make_unique<Impl>();
    impl->device     = device;
    impl->nx = dim_x; impl->ny = dim_y; impl->nz = dim_z;
    impl->fp16       = fp16;
    impl->elem_bytes = fp16 ? 2 : 4;
    impl->n          = (std::size_t)dim_x * dim_y * dim_z;

    // Import the external memory backing the image (opaque handle). Images map as a dedicated allocation.
    if (cudaImportExternalMemory(&impl->ext_mem, &mem_desc) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaImportExternalMemory failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return t;
    }

    // Map it as a single-level array. cudaArrayDefault matches a SAMPLED Vulkan image; cudaMemcpy3D
    // writes into it (the image must also carry TRANSFER_DST, which Unreal textures do).
    const cudaChannelFormatDesc channel =
        cudaCreateChannelDesc(fp16 ? 16 : 32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExternalMemoryMipmappedArrayDesc arr_desc{};
    arr_desc.offset     = mem_offset;
    arr_desc.formatDesc = channel;
    arr_desc.extent     = make_cudaExtent((std::size_t)dim_x, (std::size_t)dim_y, (std::size_t)dim_z);
    arr_desc.flags      = cudaArrayDefault;
    arr_desc.numLevels  = 1;
    if (cudaExternalMemoryGetMappedMipmappedArray(&impl->mipmap, impl->ext_mem, &arr_desc) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaExternalMemoryGetMappedMipmappedArray failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return t;
    }
    if (cudaGetMipmappedArrayLevel(&impl->level0, impl->mipmap, 0) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaGetMipmappedArrayLevel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return t;
    }

    // Persistent linear staging buffer — the model writes here; the sink relayouts it into the array.
    if (cudaMalloc(&impl->staging, impl->n * impl->elem_bytes) != cudaSuccess) {
        std::fprintf(stderr, "[rfnn::vk] cudaMalloc staging (%zu bytes) failed: %s\n",
                     impl->n * impl->elem_bytes, cudaGetErrorString(cudaGetLastError()));
        return t;
    }

    t.impl_ = std::move(impl);
    return t;
}

VulkanImageTarget VulkanImageTarget::register_image(const uint8_t device_uuid[16],
                                                    int mem_fd, std::size_t mem_size,
                                                    std::size_t mem_offset,
                                                    int dim_x, int dim_y, int dim_z, bool fp16) {
    // The FD is consumed by CUDA even on failure (POSIX external-memory semantics).
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = mem_fd;
    mem_desc.size      = mem_size;
    mem_desc.flags     = cudaExternalMemoryDedicated;   // images are dedicated allocations
    return register_imported(device_uuid, &mem_desc, mem_offset, dim_x, dim_y, dim_z, fp16);
}

VulkanImageTarget VulkanImageTarget::register_image_win32(const uint8_t device_uuid[16],
                                                          void* mem_handle, std::size_t mem_size,
                                                          std::size_t mem_offset,
                                                          int dim_x, int dim_y, int dim_z, bool fp16) {
    // Win32 handles are SHARED, not consumed — the caller keeps ownership (see the header).
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    mem_desc.handle.win32.handle = mem_handle;
    mem_desc.size                = mem_size;
    mem_desc.flags               = cudaExternalMemoryDedicated;
    return register_imported(device_uuid, &mem_desc, mem_offset, dim_x, dim_y, dim_z, fp16);
}

void VulkanImageTarget::bind_output(radfield3dnn::VolumeFieldPredictor& predictor,
                                    radfield3dnn::ModelOutput flag) {
    if (!impl_)
        throw radfield3dnn::BindingError(
            "VulkanImageTarget::bind_output: the target is empty (register_image failed) — nothing "
            "can be bound. Fall back to a host/CUDA buffer of your own.");
    // Bind the persistent staging buffer as the model output. The standard binding contract applies:
    // capacity is checked in elements against the predictor's grid, and a device/precision mismatch
    // with the session is refused with the full diagnostic.
    radfield3dnn::MemoryRef ref;
    ref.data      = impl_->staging;
    ref.bytes     = impl_->n * impl_->elem_bytes;
    ref.device    = radfield3dnn::DeviceKind::Cuda;
    ref.dtype     = impl_->fp16 ? radfield3dnn::ElementType::F16 : radfield3dnn::ElementType::F32;
    ref.device_id = impl_->device;
    predictor.bind_output_layer(flag, ref, /*convert_buffer*/ false);

    // Relayout staging → image at the tail of every predict run. The target must outlive the binding.
    Impl* impl = impl_.get();
    predictor.add_output_sink([impl]() { impl->relayout(); });
}

void VulkanImageTarget::clear(float value) {
    if (!impl_) return;
    if (cudaSetDevice(impl_->device) != cudaSuccess) return;
    if (impl_->fp16) {
        std::vector<__half> buf(impl_->n, __float2half(value));
        cudaMemcpy(impl_->staging, buf.data(), impl_->n * sizeof(__half), cudaMemcpyHostToDevice);
    } else {
        std::vector<float> buf(impl_->n, value);
        cudaMemcpy(impl_->staging, buf.data(), impl_->n * sizeof(float), cudaMemcpyHostToDevice);
    }
    impl_->relayout();
}

bool VulkanImageTarget::fill_test_pattern() {
    if (!impl_) return false;
    if (impl_->fp16) {
        std::fprintf(stderr, "[rfnn::vk] image test pattern requires an R32F target\n");
        return false;
    }
    if (cudaSetDevice(impl_->device) != cudaSuccess) return false;
    std::vector<float> ramp(impl_->n);
    for (std::size_t i = 0; i < impl_->n; ++i)
        ramp[i] = (impl_->n > 1) ? (float)i / (float)(impl_->n - 1) : 0.f;
    if (cudaMemcpy(impl_->staging, ramp.data(), impl_->n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    return impl_->relayout();
}

int VulkanImageTarget::device_index() const { return impl_ ? impl_->device : -1; }

float VulkanImageTarget::staging_max() const {
    if (!impl_ || !impl_->staging || impl_->n == 0) return 0.f;
    if (cudaSetDevice(impl_->device) != cudaSuccess) return 0.f;
    float mx = 0.f;
    if (impl_->fp16) {
        std::vector<__half> buf(impl_->n);
        if (cudaMemcpy(buf.data(), impl_->staging, impl_->n * sizeof(__half),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 0.f;
        for (const __half& h : buf) {
            const float v = __half2float(h);
            if (std::isfinite(v) && v > mx) mx = v;
        }
    } else {
        std::vector<float> buf(impl_->n);
        if (cudaMemcpy(buf.data(), impl_->staging, impl_->n * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 0.f;
        for (const float v : buf) {
            if (std::isfinite(v) && v > mx) mx = v;
        }
    }
    return mx;
}
VulkanImageTarget::operator bool() const { return impl_ != nullptr; }

}  // namespace vk
}  // namespace rfnn
