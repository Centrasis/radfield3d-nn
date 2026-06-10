cmake_minimum_required(VERSION 3.16...3.29)

# RadField3DNN. The project splits into two largely independent halves:
#   * the DEPLOYMENT runtime (libRadField3DNNDeploy) — ONNX Runtime + RadFiled3D + a GPU
#     export backend (Vulkan today). Pure C++20, no CUDA/tcnn/torch. Always built.
#   * the TRAINING/tcnn half (libRadField3DNN CUDA models + torch pybindings) — heavy,
#     needs CUDA + tiny-cuda-nn + libtorch. Gated behind RFNN_WITH_TCNN (default OFF) so the
#     deployment work configures and builds without the CUDA stack.
project(RadField3DNN LANGUAGES CXX)

option(RFNN_WITH_TCNN      "Build the tiny-cuda-nn CUDA models + torch python bindings" OFF)
option(RFNN_BACKEND_VULKAN "Build the Vulkan GPU export backend for the deployment runtime" ON)
option(BUILD_RADFIELDNN_TESTS "Build the radfield3d-nn tests" ON)
option(BUILD_EXAMPLES "Build the examples" OFF)
# Python bindings live in the tcnn half; only meaningful when that half is enabled.
option(Build_PyBindings "If the python bindings should be built" ${RFNN_WITH_TCNN})

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP24")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

SET(CMAKE_CXX_STANDARD 17)            # base standard; the deploy lib overrides to 20 below
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET TRUE)

# ──────────────────────────────────────────────────────────────────────────────
# tiny-cuda-nn / CUDA — only when RFNN_WITH_TCNN. Enabling CUDA as a language is
# deferred to here so the default (deployment-only) configure needs no CUDA toolkit.
# ──────────────────────────────────────────────────────────────────────────────
if(RFNN_WITH_TCNN)
  enable_language(CUDA)
  SET(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  # CUDA architectures are deliberately NOT hardcoded here. tiny-cuda-nn's own CMake
  # auto-detects the build machine's GPU(s) and populates TCNN_CUDA_ARCHITECTURES, which we
  # adopt for our own CUDA targets after FetchContent_MakeAvailable(fetch_tcnn).
  set(TCNN_BUILD_EXAMPLES OFF)
  set(TCNN_BUILD_BENCHMARK OFF)
  set(TCNN_BUILD_WITH_RTC ON)
  set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF)
  set(TCNN_BUILD_PRECISION_FP16 ON)
  set(TCNN_BUILD_PRECISION_FP32 ON)
  set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

  find_package(CUDAToolkit REQUIRED)

  # tiny-cuda-nn — Centrasis `tiny-cuda-nn-cu13-2` fork (carries the "static and JIT
  # compileable using CUDA 13.2" fix). Pinned at the cu13.2 tag.
  FetchContent_Declare(
    fetch_tcnn
    GIT_REPOSITORY https://github.com/Centrasis/tiny-cuda-nn-cu13-2.git
    GIT_TAG        e2b797b66e42fb4d297fe15779fa68736afd8619
  )
  FetchContent_MakeAvailable(fetch_tcnn)
  set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})
endif()

# ── RadFiled3D (always — both halves use it) ───────────────────────────────────
FetchContent_Declare(
  fetch_radfiled3d
  GIT_REPOSITORY https://github.com/Centrasis/RadFiled3D.git
  GIT_TAG        1.3.3
)
FetchContent_MakeAvailable(fetch_radfiled3d)

# ── ONNX Runtime (prebuilt) ────────────────────────────────────────────────────
# Deployable inference of EXPORTED models (TrainedModel) — independent of tcnn/torch, so a
# model trained here runs in a bare C++ runtime. Official prebuilt release (CUDA build; CPU
# EP fallback at runtime). Override the version with -DORT_VERSION=.
set(ORT_VERSION "1.26.0" CACHE STRING "ONNX Runtime version")
# Which CUDA major the prebuilt GPU package targets. ORT publishes a CUDA-13 build under a
# '_cuda13' asset suffix and the legacy CUDA-12 build with no suffix; CUDA-12 support is
# dropped from ORT 1.27.0 on. The runtime host must have the matching CUDA + cuDNN 9 (+ a
# matching TensorRT 10.x for the TensorRT EP) installed.
set(ORT_CUDA "13" CACHE STRING "CUDA major for the ONNX Runtime GPU build: 13 or 12")
if(ORT_CUDA STREQUAL "13")
  set(_ORT_CUDA_SFX "_cuda13")
else()
  set(_ORT_CUDA_SFX "")
endif()
if(WIN32)
  set(_ORT_PKG "onnxruntime-win-x64-gpu${_ORT_CUDA_SFX}-${ORT_VERSION}")
  set(_ORT_EXT "zip")
else()
  set(_ORT_PKG "onnxruntime-linux-x64-gpu${_ORT_CUDA_SFX}-${ORT_VERSION}")
  set(_ORT_EXT "tgz")
endif()
FetchContent_Declare(
  fetch_onnxruntime
  URL https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${_ORT_PKG}.${_ORT_EXT}
)
FetchContent_MakeAvailable(fetch_onnxruntime)
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION             "${fetch_onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${fetch_onnxruntime_SOURCE_DIR}/include")

# ── Vulkan (deployment GPU export backend) ─────────────────────────────────────
if(RFNN_BACKEND_VULKAN)
  find_package(Vulkan COMPONENTS glslangValidator)
  if(NOT Vulkan_FOUND)
    message(WARNING "RFNN_BACKEND_VULKAN requested but Vulkan was not found — disabling it.")
    set(RFNN_BACKEND_VULKAN OFF)
  endif()
endif()

### TORCH-SPECIFIC MSVC FLAGS (only relevant to the tcnn/torch half; MSVC-only) ###
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
    set(CMAKE_NINJA_CMCLDEPS_RC OFF)
    if(MSVC_Z7_OVERRIDE)
      set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
    endif()
    foreach(flag_var CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
      set(${flag_var} "${${flag_var}} /bigobj")
    endforeach(flag_var)
    add_compile_definitions(WIN32_LEAN_AND_MEAN)
    add_compile_definitions(_UCRT_LEGACY_INFINITY)
    add_compile_definitions(NOMINMAX)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif(MSVC)
### END MSVC FLAGS ###

# ══════════════════════════════════════════════════════════════════════════════
#  Deployment runtime — libRadField3DNNDeploy (C++20, no CUDA)
# ══════════════════════════════════════════════════════════════════════════════
set(_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Compile the deployment compute shaders to embeddable SPIR-V C headers (symbol = <name>_spv).
if(RFNN_BACKEND_VULKAN)
  file(MAKE_DIRECTORY ${_GEN_DIR})
  set(_SPV_HEADERS "")
  foreach(_shader airkerma_combine voxel_visibility)
    set(_hdr "${_GEN_DIR}/${_shader}_spv.h")
    add_custom_command(
      OUTPUT  ${_hdr}
      COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} -V --vn ${_shader}_spv
              -o ${_hdr} ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_shader}.comp
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${_shader}.comp
      COMMENT "Compiling ${_shader}.comp -> SPIR-V header"
      VERBATIM)
    list(APPEND _SPV_HEADERS ${_hdr})
  endforeach()
  add_custom_target(rfnn_shaders DEPENDS ${_SPV_HEADERS})
endif()

set(DEPLOY_SOURCES
  src/RadField3DNN/field_predictors.cpp
  src/RadField3DNN/device_radiation_field.cpp
  src/RadField3DNN/model_io.cpp)
if(RFNN_BACKEND_VULKAN)
  list(APPEND DEPLOY_SOURCES src/RadField3DNN/vk/vulkan_field.cpp)
endif()

add_library(libRadField3DNNDeploy STATIC ${DEPLOY_SOURCES})
set_target_properties(libRadField3DNNDeploy PROPERTIES
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON FOLDER "Library")
target_include_directories(libRadField3DNNDeploy PUBLIC include PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(libRadField3DNNDeploy PUBLIC libRadFiled3D onnxruntime)
if(RFNN_BACKEND_VULKAN)
  add_dependencies(libRadField3DNNDeploy rfnn_shaders)
  target_compile_definitions(libRadField3DNNDeploy PUBLIC RFNN_BACKEND_VULKAN)
  target_link_libraries(libRadField3DNNDeploy PUBLIC Vulkan::Vulkan)
endif()
set_target_properties(libRadField3DNNDeploy PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib")

# ══════════════════════════════════════════════════════════════════════════════
#  Training / tcnn half — libRadField3DNN (CUDA) + torch pybindings (gated)
# ══════════════════════════════════════════════════════════════════════════════
if(RFNN_WITH_TCNN)
  if(Build_PyBindings)
    set(Python_EXECUTABLE "$ENV{PYTHON_EXECUTABLE}" CACHE STRING "Path to the Python executable.")
    if(Python_EXECUTABLE)
      set(Python_ROOT_DIR "${Python_EXECUTABLE}" CACHE STRING "" FORCE)
    endif()
    set(PYTHON_VERSION "3.11" CACHE STRING "Python Version to use for pybind11 and conda.")
    set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
    find_package(Python ${PYTHON_VERSION} COMPONENTS Interpreter Development.Module REQUIRED)
    set(PYTHON_LIBRARY_DIR "${Python_SITELIB}")

    if(NOT DEFINED fetch_pybind_POPULATED)
      FetchContent_Declare(fetch_pybind GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v3.0.1)
      FetchContent_GetProperties(fetch_pybind)
      if(NOT fetch_pybind_POPULATED)
        FetchContent_Populate(fetch_pybind)
        add_subdirectory(${fetch_pybind_SOURCE_DIR} ${fetch_pybind_BINARY_DIR})
      endif()
    endif()

    if (NOT Torch_FOUND)
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import torch,os;from torch.utils.cpp_extension import include_paths;print(include_paths()[0] if include_paths() else '')"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(Torch_DIR "${TORCH_INCLUDE_DIR}/../share/cmake/Torch")
      find_package(Torch REQUIRED)
      find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    endif()
  endif(Build_PyBindings)

  # All CUDA sources live under src/RadField3DNN/tcnn/ (the rfnn::tcnn half); recurse so the
  # nested encodings/ layers/ utils/ models/ subfolders are picked up regardless of depth.
  file(GLOB_RECURSE CU_SOURCES "src/RadField3DNN/*.cu")
  source_group("CUDA" FILES ${CU_SOURCES})
  add_library(libRadField3DNN STATIC ${CU_SOURCES})
  set_target_properties(libRadField3DNN PROPERTIES FOLDER "Library"
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

  set(CUDA_NVCC_FLAGS "")
  if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=/bigobj")
  else()
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion" "-Xcompiler=-fno-strict-aliasing" "-Xcompiler=-fPIC")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "--extended-lambda" "--expt-relaxed-constexpr" "--use_fast_math")
  string(APPEND CMAKE_CUDA_FLAGS " -Xfatbin -compress-all")
  target_compile_options(libRadField3DNN PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

  target_include_directories(libRadField3DNN PUBLIC include)
  # Activates the rfnn::tcnn::ModelFactory section of model_io.h for the tcnn lib + bindings.
  target_compile_definitions(libRadField3DNN PUBLIC RFNN_WITH_TCNN)
  target_link_libraries(libRadField3DNN PUBLIC tiny-cuda-nn libRadFiled3D CUDA::cublas)
  set_target_properties(libRadField3DNN PROPERTIES
      ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/lib"
      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

  if(Build_PyBindings)
    add_library(libtorchtcnnhelper STATIC src/binding_utils.cu)
    set_target_properties(libtorchtcnnhelper PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON FOLDER "Library")
    target_include_directories(libtorchtcnnhelper PUBLIC include ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(libtorchtcnnhelper PUBLIC tiny-cuda-nn libRadFiled3D libRadField3DNN PRIVATE ${TORCH_LIBRARIES} Python::Module)
    target_compile_options(libtorchtcnnhelper PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

    pybind11_add_module(radfield3dnn src/python_api.cu)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    set_target_properties(radfield3dnn PROPERTIES
      CXX_VISIBILITY_PRESET "hidden" CUDA_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN YES
      CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON FOLDER "Library")
    target_compile_options(radfield3dnn PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
    set_property(TARGET radfield3dnn PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
    add_dependencies(radfield3dnn libRadField3DNN libtorchtcnnhelper tiny-cuda-nn)
    target_link_libraries(radfield3dnn PRIVATE libRadField3DNN libtorchtcnnhelper tiny-cuda-nn ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python::Module)
  endif()

  set_target_properties(tiny-cuda-nn PROPERTIES FOLDER "ThirdParty")
  set_target_properties(tiny-cuda-nn-resources PROPERTIES FOLDER "ThirdParty")
endif(RFNN_WITH_TCNN)

# ══════════════════════════════════════════════════════════════════════════════
#  Tests
# ══════════════════════════════════════════════════════════════════════════════
if(BUILD_RADFIELDNN_TESTS)
  if(NOT DEFINED fetch_googletest_POPULATED)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  enable_testing()
  include(GoogleTest)

  # Deployment tests (no CUDA): Vulkan export + model-package round-trip.
  if(RFNN_BACKEND_VULKAN)
    add_executable(test_vulkan_export tests/cxx/test_vulkan_export.cpp)
    set_target_properties(test_vulkan_export PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON FOLDER "Tests")
    target_link_libraries(test_vulkan_export libRadField3DNNDeploy GTest::gtest_main)
    gtest_discover_tests(test_vulkan_export)
  endif()

  add_executable(test_model_io tests/cxx/test_model_io.cpp)
  set_target_properties(test_model_io PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON FOLDER "Tests")
  target_compile_definitions(test_model_io PRIVATE
    RFNN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/cxx/data")
  target_link_libraries(test_model_io libRadField3DNNDeploy GTest::gtest_main)
  gtest_discover_tests(test_model_io)

  # tcnn CUDA tests (gated).
  if(RFNN_WITH_TCNN)
    add_executable(network_test tests/cxx/test_network.cu)
    target_link_libraries(network_test libRadField3DNN tiny-cuda-nn stdc++fs GTest::gtest_main)
    target_include_directories(network_test PUBLIC include)
    set_target_properties(network_test PROPERTIES FOLDER "Tests")
  endif()
endif()

if(TARGET libRadFiled3D)
  set_target_properties(libRadFiled3D PROPERTIES FOLDER "Library")
endif()
if(TARGET fmt)
  set_target_properties(fmt PROPERTIES FOLDER "ThirdParty")
endif()
if(TARGET glm)
  set_target_properties(glm PROPERTIES FOLDER "ThirdParty")
endif()
