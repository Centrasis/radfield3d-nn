#pragma once
// device_export.h — umbrella that selects the active GPU export backend at compile time.
// Today only Vulkan is implemented (RFNN_BACKEND_VULKAN); DirectML/CUDA/ROCm headers will
// be included here under their own RFNN_BACKEND_* guards and expose the same
// write_*_to_<backend> helper shape, so call sites can switch backends by a build flag.
//
// Including this header pulls in whichever backend is enabled. Code that only needs the
// backend-agnostic field should include device_radiation_field.h directly instead.

#include "radfield3d-nn/device_radiation_field.h"

#if defined(RFNN_BACKEND_VULKAN)
#  include "radfield3d-nn/vk/vulkan_field.h"
#endif

// Future:
// #if defined(RFNN_BACKEND_DIRECTML)
// #  include "radfield3d-nn/directml_field.h"
// #endif
// #if defined(RFNN_BACKEND_CUDA)
// #  include "radfield3d-nn/cuda_field.h"
// #endif
// #if defined(RFNN_BACKEND_ROCM)
// #  include "radfield3d-nn/rocm_field.h"
// #endif
