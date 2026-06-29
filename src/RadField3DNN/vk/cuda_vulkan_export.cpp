// CUDA↔Vulkan zero-copy export (NVIDIA). Imports a UE-owned external Vulkan image + timeline semaphore
// (opaque FDs) into CUDA, copies inference results into the image (cudaMemcpy3D linear→array, handling the
// optimal-tiling relayout), and signals the semaphore so UE's render can wait. See cuda_vulkan_export.h.
//
// HOST-ONLY CUDA RUNTIME — no __global__ kernels, no nvcc. Builds as a normal .cpp with the deploy lib's
// clang+libc++ toolchain (ABI-matched to UE) and links libcudart (C API). cudaMemcpy3D does the relayout,
// so no surface writes / storage-image requirement — works with UE's sampled (ShaderResource) texture.

#include "radfield3d-nn/vk/cuda_vulkan_export.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace rfnn {
namespace cuda_vk {

namespace {

bool cuda_ok(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "[cuda_vk] %s: %s\n", what, cudaGetErrorString(e));
        return false;
    }
    return true;
}

int find_device_by_uuid(const uint8_t uuid[16])
{
    int count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return -1;
    for (int d = 0; d < count; ++d)
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, d) != cudaSuccess) continue;
        if (std::memcmp(prop.uuid.bytes, uuid, 16) == 0) return d;
    }
    return -1;
}

}  // namespace

struct ImportedTarget
{
    int                     device  = -1;
    cudaExternalMemory_t    ext_mem = nullptr;
    cudaMipmappedArray_t    mipmap  = nullptr;
    cudaArray_t             level0  = nullptr;   // mip 0 — the cudaMemcpy3D destination
    cudaExternalSemaphore_t ext_sem = nullptr;
    int    nx = 0, ny = 0, nz = 0;
    bool   fp16 = false;
    size_t elem_bytes = 4;
};

bool is_available(const uint8_t device_uuid[16])
{
    return find_device_by_uuid(device_uuid) >= 0;
}

int device_index(const ImportedTarget* t)
{
    return t ? t->device : -1;
}

void* device_malloc(int dev, size_t bytes)
{
    if (!cuda_ok(cudaSetDevice(dev), "cudaSetDevice")) return nullptr;
    void* p = nullptr;
    if (!cuda_ok(cudaMalloc(&p, bytes), "cudaMalloc")) return nullptr;
    return p;
}

void device_free(int dev, void* p)
{
    if (p) { cudaSetDevice(dev); cudaFree(p); }
}

bool device_copy_d2d(void* dst, const void* src, size_t bytes)
{
    return cuda_ok(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice), "cudaMemcpy d2d");
}

ImportedTarget* import_target(const uint8_t device_uuid[16],
                              int mem_fd, size_t mem_size, size_t mem_offset,
                              int sem_fd,
                              int dim_x, int dim_y, int dim_z, bool fp16)
{
    const int device = find_device_by_uuid(device_uuid);
    if (device < 0)
    {
        std::fprintf(stderr, "[cuda_vk] no CUDA device matching the Vulkan device UUID\n");
        return nullptr;
    }
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) return nullptr;

    ImportedTarget* t = new ImportedTarget();
    t->device = device;
    t->nx = dim_x; t->ny = dim_y; t->nz = dim_z;
    t->fp16 = fp16; t->elem_bytes = fp16 ? 2 : 4;

    // Import the external memory backing UE's image (opaque FD). Images map as a dedicated allocation.
    cudaExternalMemoryHandleDesc mem_desc{};
    mem_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    mem_desc.handle.fd = mem_fd;
    mem_desc.size      = mem_size;
    mem_desc.flags     = cudaExternalMemoryDedicated;
    if (!cuda_ok(cudaImportExternalMemory(&t->ext_mem, &mem_desc), "cudaImportExternalMemory"))
    {
        destroy_target(t);
        return nullptr;
    }

    // Map it as a single-level array. cudaArrayDefault matches a SAMPLED Vulkan image (UE's ShaderResource
    // texture); cudaMemcpy3D writes into it (the image must also carry TRANSFER_DST, which UE textures do).
    const cudaChannelFormatDesc channel =
        cudaCreateChannelDesc(fp16 ? 16 : 32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExternalMemoryMipmappedArrayDesc arr_desc{};
    arr_desc.offset     = mem_offset;
    arr_desc.formatDesc = channel;
    arr_desc.extent     = make_cudaExtent((size_t)dim_x, (size_t)dim_y, (size_t)dim_z);
    arr_desc.flags      = cudaArrayDefault;
    arr_desc.numLevels  = 1;
    if (!cuda_ok(cudaExternalMemoryGetMappedMipmappedArray(&t->mipmap, t->ext_mem, &arr_desc),
                 "cudaExternalMemoryGetMappedMipmappedArray"))
    {
        destroy_target(t);
        return nullptr;
    }
    if (!cuda_ok(cudaGetMipmappedArrayLevel(&t->level0, t->mipmap, 0), "cudaGetMipmappedArrayLevel"))
    {
        destroy_target(t);
        return nullptr;
    }

    // Import UE's timeline semaphore (opaque FD) for signalling write-completion.
    cudaExternalSemaphoreHandleDesc sem_desc{};
    sem_desc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    sem_desc.handle.fd = sem_fd;
    if (!cuda_ok(cudaImportExternalSemaphore(&t->ext_sem, &sem_desc), "cudaImportExternalSemaphore"))
    {
        destroy_target(t);
        return nullptr;
    }
    return t;
}

void destroy_target(ImportedTarget* t)
{
    if (!t) return;
    if (t->device >= 0) cudaSetDevice(t->device);
    if (t->ext_sem) cudaDestroyExternalSemaphore(t->ext_sem);
    if (t->mipmap)  cudaFreeMipmappedArray(t->mipmap);   // frees the mapped array view
    if (t->ext_mem) cudaDestroyExternalMemory(t->ext_mem);
    delete t;
}

namespace {

// Copy a linear (device or host) source of N = nx*ny*nz elements into the imported array, then signal.
bool copy_and_signal(ImportedTarget* t, const void* src, cudaMemcpyKind kind, uint64_t signal_value)
{
    if (!t || !t->level0 || !t->ext_sem) return false;
    if (!cuda_ok(cudaSetDevice(t->device), "cudaSetDevice")) return false;

    cudaMemcpy3DParms p{};
    p.srcPtr = make_cudaPitchedPtr(const_cast<void*>(src),
                                   (size_t)t->nx * t->elem_bytes,   // row pitch (bytes)
                                   (size_t)t->nx,                   // width  (elements)
                                   (size_t)t->ny);                  // height (rows)
    p.dstArray = t->level0;
    p.extent   = make_cudaExtent((size_t)t->nx, (size_t)t->ny, (size_t)t->nz);  // width in ELEMENTS for arrays
    p.kind     = kind;
    if (!cuda_ok(cudaMemcpy3D(&p), "cudaMemcpy3D")) return false;

    cudaExternalSemaphoreSignalParams sig{};
    sig.params.fence.value = signal_value;
    if (!cuda_ok(cudaSignalExternalSemaphoresAsync(&t->ext_sem, &sig, 1, /*stream*/ 0),
                 "cudaSignalExternalSemaphoresAsync"))
    {
        return false;
    }
    return true;
}

}  // namespace

bool write_device_and_signal(ImportedTarget* t, const void* src_device, uint64_t signal_value)
{
    return copy_and_signal(t, src_device, cudaMemcpyDeviceToDevice, signal_value);
}

bool write_test_pattern_and_signal(ImportedTarget* t, uint64_t signal_value)
{
    if (!t) return false;
    if (t->fp16)
    {
        std::fprintf(stderr, "[cuda_vk] test pattern requires an R32F target\n");
        return false;
    }
    const size_t n = (size_t)t->nx * t->ny * t->nz;
    std::vector<float> ramp(n);
    for (size_t i = 0; i < n; ++i) ramp[i] = (n > 1) ? (float)i / (float)(n - 1) : 0.f;
    return copy_and_signal(t, ramp.data(), cudaMemcpyHostToDevice, signal_value);
}

}  // namespace cuda_vk
}  // namespace rfnn
