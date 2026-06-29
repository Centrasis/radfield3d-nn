#pragma once
// CUDA↔Vulkan zero-copy export — write ONNX (CUDA EP) inference results straight into a caller-provided
// (e.g. Unreal-owned) external Vulkan IMAGE, synchronised by an imported Vulkan timeline semaphore, with
// no host download. NVIDIA only; compiled in when RFNN_CUDA_VULKAN_INTEROP.
//
// Direction (decided with the UE integration): UE allocates the sampleable external Texture3D and exports
// its VkDeviceMemory + a timeline semaphore as opaque FDs; this module imports them into CUDA so the
// field's prediction layer IS that image's memory. CUDA copies the inference result into the image
// (cudaMemcpy3D linear→cudaArray, which handles the optimal-tiling relayout) and signals the semaphore;
// UE waits + samples.
//
// MINIMAL CUDA: this is a plain .cpp using only host-side CUDA RUNTIME calls (no __global__ kernels, no
// nvcc) — cudaMemcpy3D does the relayout. So it compiles with the SAME clang+libc++ toolchain as the rest
// of the deploy lib (ABI-matched to UE) and only links libcudart (a C API, no C++ ABI surface).
//
// The header carries NO CUDA/Vulkan types so callers (UE plugin / field_predictors) can include it freely;
// the concrete CUDA objects live behind an opaque handle in cuda_vulkan_export.cpp.

#include <cstddef>
#include <cstdint>

namespace rfnn {
namespace cuda_vk {

// True if the CUDA<->Vulkan interop is compiled in AND a CUDA device matching `device_uuid` (UE's
// VkPhysicalDevice UUID, 16 bytes) exists. Lets the caller pick this exporter or fall back to host-copy.
bool is_available(const uint8_t device_uuid[16]);

// One imported export target: an external Vulkan image (R32F or R16F, dims voxels) + a timeline semaphore,
// both imported into CUDA on the device matching `device_uuid`. Opaque; created by import_target().
struct ImportedTarget;

// Import a UE-exported external Vulkan image (opaque-FD memory) + timeline semaphore (opaque FD). The image
// is the field's prediction layer: `dim_x/y/z` voxels, `fp16` selects R16F vs R32F. `mem_size` is the whole
// allocation size; `mem_offset` the image's offset within it (from UE's RHIGetAllocationInfo). The FDs are
// consumed by CUDA (it owns/closes them). Returns nullptr on failure (then use the host-copy fallback).
ImportedTarget* import_target(const uint8_t device_uuid[16],
                              int mem_fd, size_t mem_size, size_t mem_offset,
                              int sem_fd,
                              int dim_x, int dim_y, int dim_z, bool fp16);

void destroy_target(ImportedTarget* target);

// The CUDA device index this target was imported on (matches the Vulkan device UUID). -1 if null. Pass to
// VolumeFieldPredictor::predict_to_device so ORT outputs land on the SAME device the texture lives on.
int device_index(const ImportedTarget* target);

// Minimal device-memory helpers — used to assemble a tiled per-voxel prediction into ONE device buffer
// (a voxel model queried over a volume yields the same N-value output as a volume model; each chunk's
// ONNX output is copied into the right offset). device->device copy is synchronous.
void* device_malloc(int device_index, size_t bytes);            // null on failure
void  device_free(int device_index, void* ptr);
bool  device_copy_d2d(void* dst, const void* src, size_t bytes);

// Copy `dim_x*dim_y*dim_z` scalars from a CUDA DEVICE pointer (the ONNX output tensor) into the imported
// image via cudaMemcpy3D (linear→array relayout), then signal the timeline semaphore to `signal_value` so
// UE's render can wait. `src_device` element type MUST match the image format (4 bytes if !fp16 else 2) —
// it is a straight byte copy, no conversion (the field layer + image format are kept in sync). False on failure.
bool write_device_and_signal(ImportedTarget* target, const void* src_device, uint64_t signal_value);

// Gate/self-test: fill the imported image with a deterministic ramp (value = normalized voxel index) built
// on the host and cudaMemcpy3D'd in, then signal. Lets UE read the image back and confirm CUDA wrote it.
// Requires an R32F (fp16==false) target.
bool write_test_pattern_and_signal(ImportedTarget* target, uint64_t signal_value);

}  // namespace cuda_vk
}  // namespace rfnn
