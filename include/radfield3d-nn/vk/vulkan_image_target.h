#pragma once
// A caller-owned Vulkan IMAGE registered as a prediction target: bind it for a model output and every
// predict_volume()/predict_voxels() writes the inference result into the image's memory. Unlike a
// buffer (which is linear, so the model's output simply IS the Vulkan buffer — see
// vulkan_buffer_target.h), an optimally-tiled image needs a relayout copy, so this target owns a
// persistent CUDA staging buffer that the model writes into and, on every run, relayouts into the
// image (cudaMemcpy3D linear→array). The staging buffer PERSISTS across runs, so predict_voxels()
// keeps voxels it did not touch (progressive fill).
//
// The renderer keeps full ownership: it created the image (e.g. an Unreal TexCreate_External
// Texture3D), it samples it; this type only makes it writable by the inference session. HOW that works
// is deliberately not part of the API — the header carries no CUDA/Vulkan types, and the mechanics live
// behind an opaque Impl. Compiled in when RFNN_CUDA_VULKAN_INTEROP (NVIDIA only).
//
// Completion/visibility: the post-run sink synchronises CUDA before returning, so when predict_volume()
// returns the image has been written — the renderer needs no cross-API semaphore for the write itself
// (the usual external-memory acquire barrier on its own queue is enough before sampling).

#include <cstddef>
#include <cstdint>
#include <memory>

#include <radfield3d-nn/model_interface.h>

namespace radfield3dnn { class VolumeFieldPredictor; }

namespace rfnn {
namespace vk {

class VulkanImageTarget {
public:
    // Register an existing Vulkan image from what the renderer exports:
    //   device_uuid — the VkPhysicalDevice UUID (16 bytes) the image lives on
    //   mem_fd      — its VkDeviceMemory exported as an opaque FD (vkGetMemoryFdKHR, OPAQUE_FD);
    //                 the FD is consumed by the registration (even on failure — POSIX semantics)
    //   mem_size    — the WHOLE allocation size; mem_offset — the image's offset within it
    //   dim_x/y/z   — the image's voxel counts; fp16 selects R16F (2-byte) vs R32F (4-byte) elements
    // Returns an empty target (operator bool == false) on failure — then fall back to the host path.
    static VulkanImageTarget register_image(const uint8_t device_uuid[16],
                                            int mem_fd, std::size_t mem_size, std::size_t mem_offset,
                                            int dim_x, int dim_y, int dim_z, bool fp16);

    // Windows: the same registration from an OPAQUE_WIN32 handle (vkGetMemoryWin32HandleKHR). The
    // handle stays OWNED BY THE CALLER (Win32 sharing semantics — unlike the FD, which is consumed);
    // keep it open until the target is destroyed. Fails at runtime on non-Windows.
    static VulkanImageTarget register_image_win32(const uint8_t device_uuid[16],
                                                  void* mem_handle, std::size_t mem_size,
                                                  std::size_t mem_offset,
                                                  int dim_x, int dim_y, int dim_z, bool fp16);

    // Bind this image as the destination for `flag` (Flux in practice). The predictor must execute on
    // the same device (load with Backend::Cuda and ExecutionOptions::device_id == device_index()).
    // Internally: binds an internal persistent CUDA staging buffer (dim_x*dim_y*dim_z elements) as the
    // predictor's output, and registers a post-run sink so EVERY predict_volume()/predict_voxels()
    // relayouts staging → image (cudaMemcpy3D) before returning. Synchronous: when predict returns, the
    // image is written. The target must outlive the binding (the sink reads it on every run).
    void bind_output(radfield3dnn::VolumeFieldPredictor& predictor, radfield3dnn::ModelOutput flag);

    // Fill the staging buffer (and the image) with `value` — used to reset the field to -inf on a beam
    // change so predict_voxels() then fills only the visible subset (progressive fill).
    void clear(float value);
    // Gate/self-test: fill the staging buffer + image with a deterministic ramp (value = normalized
    // voxel index) and relayout it, so the renderer can read the image back and confirm CUDA wrote it.
    // Requires an R32F (fp16 == false) target.
    bool fill_test_pattern();

    // The CUDA device ordinal matching the Vulkan device UUID — pass as ExecutionOptions::device_id so
    // the session executes where the image lives. -1 for an empty target.
    int device_index() const;

    // Max of the staging buffer's current contents (one DtoH copy + host scan; fp16-aware; non-finite
    // values ignored). The renderer auto-scales by the layer max, which the HOST path computes during
    // its upload — the zero-copy path must get it from here after a predict, or the density scale is
    // stale and the volume renders arbitrarily dim/blown-out. 0 for an empty target / on failure.
    float staging_max() const;

    explicit operator bool() const;

    VulkanImageTarget();
    ~VulkanImageTarget();
    VulkanImageTarget(VulkanImageTarget&&) noexcept;
    VulkanImageTarget& operator=(VulkanImageTarget&&) noexcept;
    VulkanImageTarget(const VulkanImageTarget&) = delete;
    VulkanImageTarget& operator=(const VulkanImageTarget&) = delete;

private:
    struct Impl;
    // Shared registration body. `handle_desc` is the platform handle descriptor, opaque here so the
    // header stays free of CUDA types (both public entry points build it in the .cpp).
    static VulkanImageTarget register_imported(const uint8_t device_uuid[16],
                                               const void* handle_desc, std::size_t mem_offset,
                                               int dim_x, int dim_y, int dim_z, bool fp16);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace rfnn
