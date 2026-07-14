#pragma once
// A caller-owned Vulkan BUFFER registered as a DIRECT prediction target: bind it for a model output
// and every predict_volume()/predict_voxels() writes the inference result into the buffer's memory —
// no copy after inference, no relayout, no host download, nothing allocated. This is the buffer
// counterpart of the image path (cuda_vulkan_export.h): an image needs a tiling relayout copy, a
// buffer is linear, so the model's output simply IS the Vulkan buffer.
//
// The renderer keeps full ownership: it created the buffer, it samples/consumes it; this type only
// makes it writable by the inference session. HOW that works is deliberately not part of the API —
// the header carries no CUDA/Vulkan types, and the mechanics live behind an opaque Impl. Compiled in
// when RFNN_CUDA_VULKAN_INTEROP (NVIDIA only; the interop needs no Vulkan SDK).
//
// Completion/visibility: predict_volume() returns only after the device writes finished, so the
// renderer needs no cross-API semaphore for the write itself — the usual external-memory acquire
// barrier on its own queue is enough before sampling. (For frame-pipelined signalling, the timeline
// semaphore import in cuda_vulkan_export.h composes with this target.)

#include <cstddef>
#include <cstdint>
#include <memory>

#include <radfield3d-nn/model_interface.h>

namespace radfield3dnn { class VolumeFieldPredictor; }

namespace rfnn {
namespace vk {

class VulkanBufferTarget {
public:
    // Register an existing Vulkan buffer from what the renderer exports:
    //   device_uuid — the VkPhysicalDevice UUID (16 bytes) the buffer lives on
    //   mem_fd      — its VkDeviceMemory exported as an opaque FD (vkGetMemoryFdKHR, OPAQUE_FD);
    //                 the FD is consumed by the registration
    //   mem_size    — the WHOLE allocation size; mem_offset — the buffer's offset within it
    //   bytes       — the buffer's usable capacity from that offset (bind_output checks it against
    //                 what the bound flag needs at the predictor's grid)
    // Returns an empty target (operator bool == false) on failure — then fall back to the host path.
    static VulkanBufferTarget register_buffer(const uint8_t device_uuid[16],
                                              int mem_fd, std::size_t mem_size,
                                              std::size_t mem_offset, std::size_t bytes);

    // Windows: the same registration from an OPAQUE_WIN32 handle (vkGetMemoryWin32HandleKHR).
    // The handle stays OWNED BY THE CALLER (Win32 sharing semantics — unlike the FD, which is
    // consumed); keep it open until the target is destroyed. Fails at runtime on non-Windows.
    static VulkanBufferTarget register_buffer_win32(const uint8_t device_uuid[16],
                                                    void* mem_handle, std::size_t mem_size,
                                                    std::size_t mem_offset, std::size_t bytes);

    // Bind this buffer as the destination for `flag`. The predictor must execute on the same device
    // (load with Backend::Cuda and ExecutionOptions::device_id == device_index(); a mismatch is
    // refused with the standard binding diagnostic). The target must outlive the binding — the
    // predictor writes into this memory on every run.
    void bind_output(radfield3dnn::VolumeFieldPredictor& predictor,
                     radfield3dnn::ModelOutput flag, bool convert_buffer = false) const;

    std::size_t capacity_bytes() const;
    // The CUDA device ordinal matching the Vulkan device UUID — pass as ExecutionOptions::device_id
    // so the session executes where the buffer lives. -1 for an empty target.
    int device_index() const;

    explicit operator bool() const;

    VulkanBufferTarget();
    ~VulkanBufferTarget();
    VulkanBufferTarget(VulkanBufferTarget&&) noexcept;
    VulkanBufferTarget& operator=(VulkanBufferTarget&&) noexcept;
    VulkanBufferTarget(const VulkanBufferTarget&) = delete;
    VulkanBufferTarget& operator=(const VulkanBufferTarget&) = delete;

private:
    struct Impl;
    // Shared registration body. `handle_desc` is the platform handle descriptor, opaque here so the
    // header stays free of CUDA types (both public entry points build it in the .cpp).
    static VulkanBufferTarget register_imported(const uint8_t device_uuid[16],
                                                const void* handle_desc,
                                                std::size_t mem_offset, std::size_t bytes);
    std::unique_ptr<Impl> impl_;
};

}  // namespace vk
}  // namespace rfnn
