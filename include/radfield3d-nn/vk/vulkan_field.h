#pragma once
// Vulkan export helpers — copy a DeviceCartesianRadiationField's per-voxel data into
// Vulkan device buffers a renderer (e.g. Unreal Engine 5 on Linux, which already owns a
// VkDevice) can sample. The caller supplies its Vulkan context; we never create a device.
//
// Three helpers, each able to either allocate a fresh device-local buffer OR write into an
// existing caller-owned VkBuffer (pass it as `target`):
//   * write_flux_to_vulkan      — the flux layer            (N floats)
//   * write_spectrum_to_vulkan  — the spectrum histogram    (N*bins floats)
//   * write_airkerma_to_vulkan  — flux+spectrum folded into per-voxel air-kerma (N floats)
//     via the airkerma_combine compute shader.
//
// This header is only needed by callers that actually export to Vulkan; the field itself
// (device_radiation_field.h) stays backend-agnostic. Compiled in when RFNN_BACKEND_VULKAN.

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "radfield3d-nn/device_radiation_field.h"

namespace rfnn {
    namespace vk {

        // Caller-owned Vulkan context. UE5 (or any host app) fills this from its own device.
        struct VulkanContext {
            VkInstance       instance        = VK_NULL_HANDLE;
            VkPhysicalDevice physical_device = VK_NULL_HANDLE;
            VkDevice         device          = VK_NULL_HANDLE;
            VkQueue          queue           = VK_NULL_HANDLE;  // a queue with COMPUTE+TRANSFER
            uint32_t         queue_family    = 0;
        };

        // A Vulkan buffer handed back to the caller. If `owns_memory` is true the helper allocated
        // it and the caller must release it via destroy(); if false it references the caller's own
        // `target` buffer (we wrote into it but do not own it).
        struct VulkanBuffer {
            VkBuffer       buffer       = VK_NULL_HANDLE;
            VkDeviceMemory memory       = VK_NULL_HANDLE;
            VkDeviceSize   size_bytes   = 0;
            uint32_t       element_count = 0;   // floats (flux/airkerma: N; spectrum: N*bins)
            bool           owns_memory  = false;
        };

        // (kPredictionChannel / kFluxLayer / kSpectrumLayer come from device_radiation_field.h.)

        // Release a buffer the helpers allocated (no-op if it does not own its memory).
        void destroy(const VulkanContext& ctx, VulkanBuffer& buf);

        // Copy `count` floats back from a device-local buffer to host memory (via a staging
        // buffer). For debugging / verification and for callers that want the result on the CPU.
        // The buffer must have been created with TRANSFER_SRC usage (the helpers' allocations are).
        std::vector<float> download_floats(const VulkanContext& ctx, VkBuffer src, uint32_t count);

        // Upload the flux layer. `target` (optional) must be DEVICE_LOCAL, usage
        // STORAGE_BUFFER|TRANSFER_DST, and large enough for N floats.
        VulkanBuffer write_flux_to_vulkan(const VulkanContext& ctx,
                                        const radfield3dnn::DeviceCartesianRadiationField& field,
                                        const std::string& channel = radfield3dnn::kPredictionChannel,
                                        VkBuffer target = VK_NULL_HANDLE);

        // Upload the spectrum histogram layer (N*bins floats, row-major per voxel).
        VulkanBuffer write_spectrum_to_vulkan(const VulkanContext& ctx,
                                            const radfield3dnn::DeviceCartesianRadiationField& field,
                                            const std::string& channel = radfield3dnn::kPredictionChannel,
                                            VkBuffer target = VK_NULL_HANDLE);

        // Combine flux+spectrum into per-voxel air-kerma on the GPU via the airkerma_combine
        // compute shader. `kerma_coeff` is the per-bin coefficient E_bin*(mu_en/rho)_bin (length
        // must equal the spectrum bin count). Result is N floats.
        VulkanBuffer write_airkerma_to_vulkan(const VulkanContext& ctx,
                                            const radfield3dnn::DeviceCartesianRadiationField& field,
                                            const std::vector<float>& kerma_coeff,
                                            const std::string& channel = radfield3dnn::kPredictionChannel,
                                            VkBuffer target = VK_NULL_HANDLE);

        // Perspective voxel-visibility cull on the GPU (compute shader, voxel_visibility.comp).
        // For a D×H×W grid, returns the normalised [0,1]^3 positions of the voxels whose centre
        // projects inside `projection` — a column-major 4×4 mapping a normalised position to clip
        // space; a voxel is kept when its projected point lies in the clip volume (x,y ∈ [-1,1],
        // z ∈ [0,1], w > 0). One-shot convenience; for a frame loop use VoxelVisibilityCuller.
        std::vector<std::array<float, 3>> compute_visible_voxels(const VulkanContext& ctx,
                                                                 std::array<int, 3> dims,
                                                                 const std::array<float, 16>& projection);

        // Cull result. The survivors live in the culler's HOST-VISIBLE buffers (valid until the
        // next cull) — no download, no copy. `positions` is tightly packed [count, 3] (x,y,z),
        // ready to bind DIRECTLY as the model's per-voxel input (zero-copy to ONNX Runtime — see
        // VoxelFieldPredictor::predict_voxelwise(const float*, ...)). `indices` is the flat voxel
        // index ((i*H+j)*W+k) per survivor, for scattering the predictions back into a field.
        struct VisibleVoxels {
            const float*    positions = nullptr;  // count*3 floats (x,y,z)
            const uint32_t* indices   = nullptr;  // count flat voxel indices
            uint32_t        count     = 0;
        };

        // Persistent voxel-visibility culler for the per-frame hot path. Builds the compute
        // pipeline ONCE (descriptor layout, shader, pipeline, pool, reusable host-visible buffers)
        // and reuses it on every cull() — construct ONE at startup, reuse it at 60fps+, destroy at
        // shutdown; do NOT rebuild per frame. Uses the caller's VulkanContext (e.g. UE5's VkDevice).
        // A UE5 RHI/RDG pipeline cannot be bound from raw Vulkan, so this owns its own VkPipeline
        // for the shader on that shared device. compute_visible_voxels() is a one-shot wrapper.
        class VoxelVisibilityCuller {
        public:
            explicit VoxelVisibilityCuller(const VulkanContext& ctx);
            ~VoxelVisibilityCuller();
            VoxelVisibilityCuller(const VoxelVisibilityCuller&) = delete;
            VoxelVisibilityCuller& operator=(const VoxelVisibilityCuller&) = delete;

            // Cull one frame. The GPU writes the survivors into host-visible memory; only the
            // 4-byte count is read on the host. Returns pointers INTO the culler's buffers (valid
            // until the next cull) — zero-copy hand-off, no explicit download.
            VisibleVoxels cull(std::array<int, 3> dims, const std::array<float, 16>& projection);

            struct Impl;
        private:
            std::unique_ptr<Impl> impl_;
        };

    } // namespace vk
}  // namespace rfnn
