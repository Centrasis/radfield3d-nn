// Capability probe: does any ONNX Runtime execution provider on THIS machine/package support
// zero-copy import of a Vulkan buffer + Vulkan timeline semaphore (ORT 1.24+ Interop API)?
//
// This is the single datapoint that decides the architecture for "ONNX writes straight into a
// UE-owned VkBuffer, no copy":
//   * If an importer reports CanImportMemory(VK_MEMORY_OPAQUE_FD) == true, the clean path exists —
//     UE exports its VkBuffer/semaphore as opaque FDs, ORT imports them and binds the model outputs
//     as tensors over that memory (CreateTensorFromMemory -> BindOutput). No CUDA interop code.
//   * If NO importer supports it (the shipped CUDA package only ships cuda/tensorrt/shared providers;
//     Vulkan import is documented as tied to the NvTensorRtRtx provider), the clean path is absent
//     and the fallback is hand-written CUDA<->Vulkan external-memory interop.
//
// Build (against the already-fetched ORT in the plugin build dir), then run on the GPU box:
//   ORT=<plugin>/Source/ThirdParty/RadField3DNNLibrary/build/_deps/fetch_onnxruntime-src
//   g++ -std=c++17 probe_vulkan_interop.cpp -I"$ORT/include" -L"$ORT/lib" -lonnxruntime \
//       -Wl,-rpath,"$ORT/lib" -o probe_vulkan_interop
//   ./probe_vulkan_interop
//
// To test the TensorRT-RTX package instead, point ORT at that package's src dir and re-run.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <onnxruntime_c_api.h>

static const OrtApi* g = nullptr;

static void check(OrtStatus* st, const char* what) {
    if (st) { std::fprintf(stderr, "FAIL %s: %s\n", what, g->GetErrorMessage(st)); g->ReleaseStatus(st); }
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: survive a crash inside ORT registration
    g = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g) { std::fprintf(stderr, "no OrtApi for ORT_API_VERSION=%d\n", ORT_API_VERSION); return 2; }

    const OrtInteropApi* interop = g->GetInteropApi();
    std::printf("GetInteropApi(): %s\n", interop ? "present" : "NULL (runtime too old / no interop)");
    if (!interop) return 1;

    OrtEnv* env = nullptr;
    check(g->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "probe", &env), "CreateEnv");
    if (!env) return 2;

    // Classic provider list (what the package ships) — proves CUDA/TensorRT are present even if they
    // do not surface as autoEP OrtEpDevices below.
    char** provs = nullptr; int nprov = 0;
    if (!g->GetAvailableProviders(&provs, &nprov)) {
        std::printf("Classic providers in this package:");
        for (int i = 0; i < nprov; ++i) std::printf(" %s", provs[i]);
        std::printf("\n");
        g->ReleaseAvailableProviders(provs, nprov);
    }

    // Try to register the bundled CUDA/TensorRT provider .so as autoEP libraries so GetEpDevices can
    // enumerate them (the importer capability is queried per OrtEpDevice). Best-effort: if the classic
    // shared provider is not a registrable autoEP plugin, this errors and we proceed with what we have.
    if (const char* dir = std::getenv("ORT_PROVIDER_DIR")) {
        struct { const char* name; const char* file; } libs[] = {
            {"cuda_ep",   "libonnxruntime_providers_cuda.so"},
            {"tensorrt_ep","libonnxruntime_providers_tensorrt.so"},
        };
        for (auto& L : libs) {
            std::string path = std::string(dir) + "/" + L.file;
            OrtStatus* st = g->RegisterExecutionProviderLibrary(env, L.name, path.c_str());
            std::printf("RegisterExecutionProviderLibrary(%s): %s\n", L.file,
                        st ? g->GetErrorMessage(st) : "ok");
            if (st) g->ReleaseStatus(st);
        }
    } else {
        std::printf("(set ORT_PROVIDER_DIR=<.../lib> to attempt autoEP registration of cuda/tensorrt)\n");
    }

    const OrtEpDevice* const* devices = nullptr;
    size_t n = 0;
    check(g->GetEpDevices(env, &devices, &n), "GetEpDevices");
    std::printf("EP devices visible: %zu\n", n);

    bool any_vk_mem = false;
    for (size_t i = 0; i < n; ++i) {
        const OrtEpDevice* dev = devices[i];
        const char* name = g->EpDevice_EpName(dev);
        const OrtHardwareDevice* hw = g->EpDevice_Device(dev);
        OrtHardwareDeviceType type = hw ? g->HardwareDevice_Type(hw) : OrtHardwareDeviceType_CPU;
        const char* tname = type == OrtHardwareDeviceType_GPU ? "GPU"
                          : type == OrtHardwareDeviceType_NPU ? "NPU" : "CPU";

        OrtExternalResourceImporter* imp = nullptr;
        check(interop->CreateExternalResourceImporterForDevice(dev, &imp), "CreateExternalResourceImporterForDevice");
        if (!imp) {
            std::printf("  [%zu] EP=%-16s dev=%-3s  importer: NONE (EP has no external-resource import)\n",
                        i, name ? name : "?", tname);
            continue;
        }

        bool mem = false, sem = false;
        check(interop->CanImportMemory(imp, ORT_EXTERNAL_MEMORY_HANDLE_TYPE_VK_MEMORY_OPAQUE_FD, &mem),
              "CanImportMemory");
        check(interop->CanImportSemaphore(imp, ORT_EXTERNAL_SEMAPHORE_VK_TIMELINE_SEMAPHORE_OPAQUE_FD, &sem),
              "CanImportSemaphore");
        std::printf("  [%zu] EP=%-16s dev=%-3s  importer: YES   VK_MEMORY_OPAQUE_FD=%s  VK_TIMELINE_SEMAPHORE=%s\n",
                    i, name ? name : "?", tname, mem ? "YES" : "no", sem ? "YES" : "no");
        any_vk_mem |= mem;
        interop->ReleaseExternalResourceImporter(imp);
    }

    std::printf("\n==> Vulkan zero-copy import available on this package: %s\n",
                any_vk_mem ? "YES (clean path is viable)" : "NO (fallback = CUDA<->Vulkan interop)");
    g->ReleaseEnv(env);
    return any_vk_mem ? 0 : 1;
}
