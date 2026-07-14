#include "radfield3d-nn/field_predictors.h"

#include "radfield3d-nn/device_radiation_field.h"
#if RFNN_CUDA_VULKAN_INTEROP
#include "radfield3d-nn/vk/cuda_vulkan_export.h"   // device_malloc/free/copy_d2d (assemble tiled output)
#endif
#include "radfield3d-nn/model_io.h"

#include <onnxruntime_cxx_api.h>

#include <RadFiled3D/Voxel.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace radfield3dnn {

namespace {
// Lower-case substring match — the Python exports use varied input names
// (e.g. "position"/"pos", "direction"/"dir", "spectrum"/"sp"), so bind by intent.
bool name_is(const std::string& n, std::initializer_list<const char*> keys) {
    std::string l = n; std::transform(l.begin(), l.end(), l.begin(), ::tolower);
    for (auto* k : keys) if (l.find(k) != std::string::npos) return true;
    return false;
}
}  // namespace

struct VolumeFieldPredictor::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "radfield3dnn"};
    Ort::SessionOptions opts;
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo mem{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    std::vector<std::string> in_names, out_names;
    std::vector<std::vector<int64_t>> in_shapes;   // graph-declared shapes (dyn dims < 0)
    // The ACTUAL output shapes. The graphs come out of torch's dynamo exporter with no declared
    // output rank (ONNX Runtime reports rank 0), so the shapes cannot be read from the session — they
    // are captured from one probe Run and cached. Pre-binding caller memory needs them; the
    // allocating path never did, because it inspects the results it gets back.
    std::vector<std::vector<int64_t>> out_shapes;
    bool out_layout_resolved = false;
    std::string provider = "CPU";  // EP that actually registered (set by configure_options): TensorRT/CUDA/CPU
    // Metric [min,max] ranges per beam-parameter name (from the RF3M ModelDomain). Used to
    // clip+normalise the inputs the model trained on in normalised form (e.g. "distance").
    std::map<std::string, std::array<float, 2>> param_ranges;

    // ── caller-owned zero-copy bindings (bind_global_parameter / bind_output_layer) ──────────────
    // One slot per interface flag, indexed by the flag's bit position.
    struct Slot {
        MemoryRef          mem;
        bool               bound   = false;
        bool               convert = false;   // staged + converted instead of bound directly
        std::vector<float> staging;           // convert only: inputs snapshot here, outputs land here
    };
    Slot in_slots[32];
    Slot out_slots[32];
    // Invoked once at the tail of every predict path (see flush_converted_outputs), after the bound
    // outputs are written & synchronised. rfnn::vk::VulkanImageTarget uses this to relayout its CUDA
    // staging buffer into the bound Vulkan image on each run. Empty on a plain host/CUDA session.
    std::vector<std::function<void()>> output_sinks;
    std::array<int, 3> grid{0, 0, 0};
    bool grid_set = false;
    int  device_id = 0;                       // the EP's device ordinal (ExecutionOptions)

    // Device-resident scratch for a GPU session (empty on a CPU session). `cuda_mem` names the
    // session's GPU so an unrequested output can be scratch-allocated on the device (no host
    // download) and so the per-voxel bound path can keep its latent + position grid device-resident.
    bool           on_device = false;
    Ort::MemoryInfo cuda_mem{nullptr};

    // ORT-CUDA-allocated persistent buffers used by the bound per-voxel path on a GPU session. They
    // are reused across runs (the allocation, not the values): `dev_positions` holds the full [N,3]
    // normalised grid, refilled only when the grid changes; `dev_latent` holds the [CH, L] broadcast
    // of the beam latent, refilled every run so an in-place edit of a bound input is seen. Device
    // pointers into these back per-chunk sub-view tensors. Left null until first use / when the device
    // data-transfer is unavailable (device_broadcast_ok == 0), in which case the bound path stays
    // host-side. `shared_cuda_alloc` is owned by the env (registered EP library) — never released here.
    OrtAllocator*  shared_cuda_alloc = nullptr;
    Ort::Value     dev_positions{nullptr};
    std::array<int, 3> dev_positions_grid{0, 0, 0};
    Ort::Value     dev_latent{nullptr};
    int            dev_latent_rows = 0;
    int            dev_latent_width = 0;
    int            device_broadcast_ok = -1;   // tri-state: -1 unprobed, 0 unavailable, 1 usable
};

// Append the TensorRT EP (V2 API). Engines are compiled per ONNX subgraph + input shape and
// cached on disk (`cache_dir`), so only the first run on a given model/shape/GPU pays the
// build. Best-effort: a non-TRT runtime (provider lib or TensorRT libs missing) logs and
// returns so the CUDA/CPU fallback appended next still serves the model.
static void append_tensorrt(VolumeFieldPredictor::Impl& im, const ExecutionOptions& exec) {
    std::string cache = exec.engine_cache_dir;
    if (cache.empty())
        cache = (std::filesystem::temp_directory_path() / "rfnn_trt_cache").string();
    std::error_code ec; std::filesystem::create_directories(cache, ec);

    const OrtApi& api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2* trt = nullptr;
    if (!Ort::Status(api.CreateTensorRTProviderOptions(&trt)).IsOK() || trt == nullptr) {
        std::fprintf(stderr, "[radfield3dnn] TensorRT EP unavailable; using CUDA/CPU.\n");
        return;
    }
    // Release the opaque options object however we leave this function.
    std::unique_ptr<OrtTensorRTProviderOptionsV2, void(*)(OrtTensorRTProviderOptionsV2*)>
        guard(trt, [](OrtTensorRTProviderOptionsV2* p) { Ort::GetApi().ReleaseTensorRTProviderOptions(p); });

    const std::string dev = std::to_string(exec.device_id);
    const std::string fp16 = exec.fp16 ? "1" : "0";
    const char* keys[] = {"device_id", "trt_fp16_enable", "trt_engine_cache_enable",
                          "trt_engine_cache_path", "trt_timing_cache_enable", "trt_timing_cache_path"};
    const char* vals[] = {dev.c_str(), fp16.c_str(), "1", cache.c_str(), "1", cache.c_str()};
    if (!Ort::Status(api.UpdateTensorRTProviderOptions(trt, keys, vals, 6)).IsOK()) {
        std::fprintf(stderr, "[radfield3dnn] failed to set TensorRT options; using CUDA/CPU.\n");
        return;
    }
    // Run on the caller's CUDA stream when one was supplied (setting the pointer option also flips the
    // EP's has_user_compute_stream flag), so no cross-stream sync is needed with the caller's buffers.
    if (exec.user_compute_stream &&
        !Ort::Status(api.UpdateTensorRTProviderOptionsWithValue(
             trt, "user_compute_stream", exec.user_compute_stream)).IsOK())
        std::fprintf(stderr, "[radfield3dnn] TensorRT EP ignored user_compute_stream.\n");
    try {
        im.opts.AppendExecutionProvider_TensorRT_V2(*trt);  // throws if EP not registrable
        im.provider = "TensorRT";
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[radfield3dnn] TensorRT EP not registrable (%s); using CUDA/CPU.\n", e.what());
    }
}

// Shared session setup: select execution providers in priority order — TensorRT (if asked),
// then CUDA as a fallback for any subgraph TRT did not claim, then CPU — and set graph
// optimization. The session itself is created by the calling ctor (from a path or a memory
// buffer), since ORT has distinct Session constructors for each.
static void configure_options(VolumeFieldPredictor::Impl& im, const ExecutionOptions& exec) {
    im.opts.SetIntraOpNumThreads(0);
    im.opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    im.device_id = exec.device_id;
    if (!exec.use_gpu) return;  // CPU only

    if (exec.use_tensorrt) append_tensorrt(im, exec);

    // CUDA EP: claims any op TRT left behind, and is the GPU path when TRT is off/absent.
    try {
        OrtCUDAProviderOptions cuda{}; cuda.device_id = exec.device_id;
        if (exec.user_compute_stream) {          // enqueue on the caller's stream (no cross-stream sync)
            cuda.has_user_compute_stream = 1;
            cuda.user_compute_stream     = exec.user_compute_stream;
        }
        im.opts.AppendExecutionProvider_CUDA(cuda);
        if (im.provider != "TensorRT") im.provider = "CUDA";   // GPU (keep TensorRT if it registered first)
    }
    catch (const std::exception&) { /* no CUDA EP -> CPU fallback (im.provider stays "CPU") */ }

    // On a GPU session, name the device once so unrequested outputs can be scratch-allocated there
    // (no host download) and the bound per-voxel path can keep its latent + positions device-resident.
    im.on_device = (im.provider != "CPU");
    if (im.on_device)
        im.cuda_mem = Ort::MemoryInfo("Cuda", OrtDeviceAllocator, exec.device_id, OrtMemTypeDefault);
}

std::string VolumeFieldPredictor::execution_provider() const { return impl_->provider; }
bool VolumeFieldPredictor::uses_gpu() const { return impl_->provider != "CPU"; }

VolumeFieldPredictor::VolumeFieldPredictor(const std::string& onnx_path, bool use_cuda)
    : VolumeFieldPredictor(onnx_path, ExecutionOptions{.use_gpu = use_cuda}) {}

VolumeFieldPredictor::VolumeFieldPredictor(const void* onnx_bytes, size_t n, bool use_cuda)
    : VolumeFieldPredictor(onnx_bytes, n, ExecutionOptions{.use_gpu = use_cuda}) {}

VolumeFieldPredictor::VolumeFieldPredictor(const std::string& onnx_path, const ExecutionOptions& exec)
    : impl_(std::make_unique<Impl>()) {
    configure_options(*impl_, exec);
    // ORT model paths are ORTCHAR_T* (wchar_t on Windows, char on POSIX); std::filesystem::path
    // yields the correct character type per platform. The temporary lives for the full expression,
    // so the pointer stays valid through Session construction.
    const std::filesystem::path p(onnx_path);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, p.c_str(), impl_->opts);
    introspect();
}

VolumeFieldPredictor::VolumeFieldPredictor(const void* onnx_bytes, size_t n, const ExecutionOptions& exec)
    : impl_(std::make_unique<Impl>()) {
    configure_options(*impl_, exec);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, onnx_bytes, n, impl_->opts);
    introspect();
}

VolumeFieldPredictor::VolumeFieldPredictor(VolumeFieldPredictor&&) noexcept = default;

void VolumeFieldPredictor::introspect() {
    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < impl_->session->GetInputCount(); ++i) {
        impl_->in_names.emplace_back(impl_->session->GetInputNameAllocated(i, alloc).get());
        impl_->in_shapes.push_back(
            impl_->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }
    for (size_t i = 0; i < impl_->session->GetOutputCount(); ++i)
        impl_->out_names.emplace_back(impl_->session->GetOutputNameAllocated(i, alloc).get());

    // A per-voxel model has a per-point coordinate ("position") input; a field-wise
    // model takes only beam parameters and emits the whole volume.
    for (const auto& n : impl_->in_names)
        if (name_is(n, {"position", "pos", "query", "xyz", "location", "loc"})) voxelwise_ = true;

    // Input beam-spectrum length: the trailing dim of the "spectrum" input (the model's required
    // tube-spectrum histogram size — distinct from the OUTPUT per-voxel histogram bins below).
    for (size_t i = 0; i < impl_->in_names.size(); ++i) {
        if (name_is(impl_->in_names[i], {"spectrum", "spec", "sp"})) {
            const auto& s = impl_->in_shapes[i];
            if (!s.empty() && s.back() > 1) in_spectrum_bins_ = static_cast<int>(s.back());
            break;
        }
    }

    // Spectrum bin count from the largest spectrum-shaped output's last dim (>=2). Also detect fp16:
    // if any output tensor is FLOAT16 the model predicts in half precision, so predict_into_field
    // builds the field's flux layer as fp16 (RadFiled3D float16) rather than float32.
    for (size_t i = 0; i < impl_->out_names.size(); ++i) {
        auto info = impl_->session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto s = info.GetShape();
        impl_->out_shapes.push_back(s);
        if (!s.empty() && s.back() > 1 && s.back() <= 1024) out_bins_ = static_cast<int>(s.back());
        if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) out_fp16_ = true;
    }
}

VolumeFieldPredictor::~VolumeFieldPredictor() = default;

SpectrumInputLayout VolumeFieldPredictor::input_spectrum_layout() const {
    SpectrumInputLayout L;
    for (const auto& bp : domain_.beam_parameters) {
        if (bp.name == "spectrum" && bp.range.type == rfnn::io::ParameterRangeType::Spectrum) {
            float scale = 1.f;   // -> eV
            if (bp.range.unit == "keV") scale = 1e3f;
            else if (bp.range.unit == "MeV") scale = 1e6f;
            L.min_energy_ev = bp.range.min * scale;
            L.max_energy_ev = bp.range.max * scale;
            L.bin_width_ev  = bp.range.bin_width * scale;
            L.bins = (L.bin_width_ev > 0.f)
                ? static_cast<int>(std::lround((L.max_energy_ev - L.min_energy_ev) / L.bin_width_ev)) : 0;
            break;
        }
    }
    return L;
}

void VolumeFieldPredictor::set_parameter_range(const std::string& name, float min, float max) {
    impl_->param_ranges[name] = {min, max};
}

void VolumeFieldPredictor::set_package_metadata(rfnn::io::ModelDomain domain,
                                                rfnn::io::ModelProvenance provenance,
                                                std::map<std::string, float> metrics,
                                                std::vector<std::string> graph_names,
                                                radfield3dnn::ModelInterface interface) {
    interface_ = interface;
    domain_      = std::move(domain);
    provenance_  = std::move(provenance);
    metrics_     = std::move(metrics);
    graph_names_ = std::move(graph_names);
}

static bool is_position_input(const std::string& n) {
    return name_is(n, {"position", "pos", "query", "xyz"});
}

// Clip `v` to a registered metric range and map it to [0,1]; if no range is registered for `name`
// the value passes through unchanged (a degenerate min==max range also passes through, mapping to 0).
static float normalize_metric(const std::string& name, float v,
                              const std::map<std::string, std::array<float, 2>>& ranges) {
    auto it = ranges.find(name);
    if (it == ranges.end()) return v;
    const float lo = it->second[0], hi = it->second[1];
    if (hi - lo <= 1e-12f) return 0.0f;
    const float c = std::min(std::max(v, lo), hi);
    return (c - lo) / (hi - lo);
}

// Bind one tensor for a *beam-parameter* graph input `n`, broadcasting the beam over `rows`
// rows. Returns false for the per-point position input (the caller binds that); throws for
// an input it does not recognize. `dist` is the precomputed (metric) source distance; `ranges`
// holds the ModelDomain [min,max] per parameter for the metric inputs the model trained on in
// normalised form ("distance", "opening_angle"/"rect").
static bool make_beam_input(const std::string& n, int rows, const BeamParameters& beam, float dist,
                     const std::map<std::string, std::array<float, 2>>& ranges,
                     Ort::MemoryInfo& mem, std::vector<std::vector<float>>& buffers,
                     std::vector<Ort::Value>& inputs) {
    auto make = [&](std::vector<float>&& data, std::vector<int64_t> shape) {
        buffers.emplace_back(std::move(data));
        inputs.emplace_back(Ort::Value::CreateTensor<float>(
            mem, buffers.back().data(), buffers.back().size(), shape.data(), shape.size()));
    };
    if (is_position_input(n)) return false;
    if (name_is(n, {"direction", "dir"})) {
        std::vector<float> d(static_cast<size_t>(rows) * 3);
        for (int i = 0; i < rows; ++i) std::memcpy(&d[3*i], beam.direction.data(), 3*sizeof(float));
        make(std::move(d), {rows, 3});
    } else if (name_is(n, {"distance"})) {
        // metric source distance -> clip to the model's [min,max]m and map to [0,1] (matches the
        // training-time BeamParametersNormalization; without it the trunk gets an out-of-range value).
        make(std::vector<float>(rows, normalize_metric("distance", dist, ranges)), {rows, 1});
    } else if (name_is(n, {"origin", "src"})) {
        std::vector<float> d(static_cast<size_t>(rows) * 3);
        for (int i = 0; i < rows; ++i) std::memcpy(&d[3*i], beam.origin.data(), 3*sizeof(float));
        make(std::move(d), {rows, 3});
    } else if (name_is(n, {"spectrum", "spec", "sp"})) {
        const int S = static_cast<int>(beam.spectrum.size());
        std::vector<float> d(static_cast<size_t>(rows) * S);
        for (int i = 0; i < rows; ++i) std::memcpy(&d[static_cast<size_t>(i)*S], beam.spectrum.data(), S*sizeof(float));
        make(std::move(d), {rows, S});
    } else if (name_is(n, {"rect", "shape", "beam_shape"})) {
        std::vector<float> d(static_cast<size_t>(rows) * 2);
        for (int i = 0; i < rows; ++i) std::memcpy(&d[2*i], beam.rect.data(), 2*sizeof(float));
        make(std::move(d), {rows, 2});
    } else {
        throw std::runtime_error("TrainedModel: unmapped ONNX input '" + n + "'");
    }
    return true;
}

static float source_distance(const BeamParameters& beam) {
    // Source-to-isocentre distance. Voxel positions run [0,1], so the ISOCENTRE is the cube centre
    // (0.5,0.5,0.5); `origin` is the source position in that same [0,1] field space (it may lie outside the
    // cube — the source is usually outside the reconstructed region). Distance = |origin − 0.5|, then
    // clipped+normalised to the model's metric range.
    return std::sqrt((beam.origin[0]-0.5f)*(beam.origin[0]-0.5f) +
                     (beam.origin[1]-0.5f)*(beam.origin[1]-0.5f) +
                     (beam.origin[2]-0.5f)*(beam.origin[2]-0.5f));
}

static std::vector<Ort::Value> run_graph(VolumeFieldPredictor::Impl& im, std::vector<Ort::Value>& inputs) {
    std::vector<const char*> in_c, out_c;
    for (auto& n : im.in_names)  in_c.push_back(n.c_str());
    for (auto& n : im.out_names) out_c.push_back(n.c_str());
    return im.session->Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(),
                           inputs.size(), out_c.data(), out_c.size());
}

// Per-voxel query: bind the position input DIRECTLY over the caller's contiguous [rows,3] (x,y,z)
// buffer (no copy — e.g. the host-visible output of vulkan::VoxelVisibilityCuller, for a zero-copy
// GPU-cull → ONNX hand-off), plus either the broadcast beam parameters (single-graph model) or the
// broadcast pre-computed latent (when the beam was encoded by a separate beam-encoder graph). The
// `positions_xyz` buffer must stay alive until Run() returns.
static std::vector<Ort::Value> run_positions(VolumeFieldPredictor::Impl& im,
                                      const float* positions_xyz, int rows,
                                      const EncodedBeam& beam, int device_id = -1) {
    const bool prof = std::getenv("RFNN_PROFILE") != nullptr;
    const auto tp0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> buffers;  // own the (beam/latent) data until Run() returns
    std::vector<Ort::Value> inputs;
    buffers.reserve(im.in_names.size());

    for (const auto& n : im.in_names) {
        if (is_position_input(n)) {
            const int64_t shape[2] = {rows, 3};
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                im.mem, const_cast<float*>(positions_xyz), static_cast<size_t>(rows) * 3, shape, 2));
        } else if (beam.is_encoded) {
            // The trunk's non-position input is the latent, broadcast over `rows`. Materialising this [rows,L]
            // buffer dominated the per-inference cost (~85 MB at 48³): a fresh per-call allocation page-faults
            // the whole region (the kernel zero-fills each page on first touch) — ~27 ms, independent of the EP.
            // So REUSE one thread-local buffer across calls (pages stay resident -> no faults after warm-up) and
            // fill by GEOMETRIC DOUBLING (~log2(rows) large memcpys, not `rows` tiny ones). The Ort::Value just
            // views this buffer; it stays valid through Run() (serial use per predictor thread). NOTE: each
            // calling thread retains this buffer at its high-water size (~rows*L*4 B); that is intentional reuse,
            // not a leak — only inference threads ever allocate it, and they reuse it every frame.
            const int L = static_cast<int>(beam.latent.size());
            const size_t total = static_cast<size_t>(rows) * L;
            static thread_local std::vector<float> bcast;
            if (bcast.size() < total) bcast.resize(total);
            float* d = bcast.data();
            std::memcpy(d, beam.latent.data(), static_cast<size_t>(L) * sizeof(float));
            for (size_t filled = 1; filled < static_cast<size_t>(rows); ) {
                const size_t copy_rows = std::min(filled, static_cast<size_t>(rows) - filled);
                std::memcpy(d + filled * L, d, copy_rows * static_cast<size_t>(L) * sizeof(float));
                filled += copy_rows;
            }
            const int64_t shape[2] = {rows, L};
            inputs.emplace_back(Ort::Value::CreateTensor<float>(im.mem, d, total, shape, 2));
        } else {
            make_beam_input(n, rows, beam.beam, source_distance(beam.beam), im.param_ranges,
                            im.mem, buffers, inputs);
        }
    }
#if RFNN_CUDA_VULKAN_INTEROP
    if (device_id >= 0) {
        // Device-only: bind the outputs to CUDA device memory so ORT writes the chunk on the GPU.
        const auto tpb = std::chrono::high_resolution_clock::now();
        Ort::MemoryInfo cuda_mem("Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
        Ort::IoBinding binding(*im.session);
        for (size_t i = 0; i < im.in_names.size(); ++i) binding.BindInput(im.in_names[i].c_str(), inputs[i]);
        for (const auto& on : im.out_names) binding.BindOutput(on.c_str(), cuda_mem);
        const auto tp1 = std::chrono::high_resolution_clock::now();
        im.session->Run(Ort::RunOptions{nullptr}, binding);
        if (prof) {
            const auto tp2 = std::chrono::high_resolution_clock::now();
            std::fprintf(stderr, "[prof] run_positions rows=%d  build=%.3f ms  bind=%.3f ms  Run=%.3f ms\n", rows,
                std::chrono::duration<double, std::milli>(tpb - tp0).count(),
                std::chrono::duration<double, std::milli>(tp1 - tpb).count(),
                std::chrono::duration<double, std::milli>(tp2 - tp1).count());
        }
        return binding.GetOutputValues();
    }
#else
    (void)device_id;
#endif
    return run_graph(im, inputs);
}

// Whole-field query: a field-wise model takes only the beam params (a single row) and emits
// the entire volume in one Run(). (Per-voxel models reach the volume via run_positions.)
static std::vector<Ort::Value> run_field(VolumeFieldPredictor::Impl& im, const BeamParameters& beam) {
    const float dist = source_distance(beam);
    std::vector<std::vector<float>> buffers;
    std::vector<Ort::Value> inputs;
    buffers.reserve(im.in_names.size());

    for (const auto& n : im.in_names) {
        if (is_position_input(n))
            throw std::runtime_error("run_field: model expects a per-point '" + n +
                                     "' input — use predict_voxelwise/predict_volume");
        make_beam_input(n, /*rows=*/1, beam, dist, im.param_ranges, im.mem, buffers, inputs);
    }
    return run_graph(im, inputs);
}

// Copy the (flux, spectrum) outputs of a Run() straight into caller memory: `valid_rows` flux
// values to `flux_dst` and `valid_rows * n_bins` spectrum values to `spec_dst` (either may be
// null to skip). Writing into a caller buffer (e.g. a DeviceCartesianRadiationField layer)
// avoids the intermediate std::vector copies. The first non-spectrum output is taken as flux;
// outputs are identified by their trailing dim (== n_bins => spectrum).
// Copy a model output tensor's `count` values into `dst` as float, converting from fp16 when the
// tensor is FLOAT16 (ONNX Runtime does not auto-convert). float32 tensors are a straight memcpy.
static void copy_tensor_floats(const Ort::Value& o, float* dst, size_t count) {
    if (!dst) return;
    if (o.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(o.GetTensorData<Ort::Float16_t>());
        for (size_t i = 0; i < count; ++i) {
            RadFiled3D::Typing::float16 v; std::memcpy(&v, &h[i], sizeof(uint16_t));
            dst[i] = static_cast<float>(v);
        }
    } else {
        std::memcpy(dst, o.GetTensorData<float>(), count * sizeof(float));
    }
}

static void collect(std::vector<Ort::Value>& outs, int n_bins, size_t valid_rows,
                    float* flux_dst, float* spec_dst) {
    bool got_flux = false;
    for (auto& o : outs) {
        auto shp = o.GetTensorTypeAndShapeInfo().GetShape();
        const bool is_spec = !shp.empty() && shp.back() == n_bins;
        if (is_spec) {
            copy_tensor_floats(o, spec_dst, valid_rows * static_cast<size_t>(n_bins));
        } else if (!got_flux) {
            copy_tensor_floats(o, flux_dst, valid_rows);
            got_flux = true;
        }
    }
}

// Ensure `out`'s prediction channel + flux (scalar) / spectrum (HistogramVoxel<float>) layers
// exist, and return their contiguous base pointers (the destinations collect() writes into).
static std::shared_ptr<DeviceVoxelBuffer> ensure_pred_layers(DeviceCartesianRadiationField& out,
                                                             int bins, bool fp16) {
    auto channel = out.has_channel(kPredictionChannel)
                       ? out.get_channel(kPredictionChannel)
                       : std::static_pointer_cast<DeviceVoxelBuffer>(out.add_channel(kPredictionChannel));
    if (!channel->has_layer(kFluxLayer)) {
        if (fp16) channel->add_layer<RadFiled3D::Typing::float16>(kFluxLayer, RadFiled3D::Typing::float16(0.f), "flux");
        else      channel->add_layer<float>(kFluxLayer, 0.f, "flux");
    }
    // Spectrum stays a float HistogramVoxel — RadFiled3D's histogram (de)serialiser is float-only.
    if (!channel->has_layer(kSpectrumLayer))
        channel->add_custom_layer<RadFiled3D::HistogramVoxel<float>, float>(
            kSpectrumLayer, RadFiled3D::HistogramVoxel<float>(static_cast<size_t>(bins), 1.f, nullptr), 0.f, "spectrum");
    return channel;
}

std::vector<float> VolumeFieldPredictor::run_field_raw(const BeamParameters& beam) const {
    // Run the (field) graph on the beam parameters and return the first output flat — used when
    // this predictor is the beam encoder of a VoxelFieldPredictor (beam parameters -> latent).
    std::vector<Ort::Value> outs = run_field(*impl_, beam);
    if (outs.empty()) throw std::runtime_error("run_field_raw: graph produced no output");
    auto shp = outs.front().GetTensorTypeAndShapeInfo().GetShape();
    size_t count = 1; for (auto d : shp) count *= (d > 0 ? static_cast<size_t>(d) : 1);
    std::vector<float> v(count);
    copy_tensor_floats(outs.front(), v.data(), count);  // fp16-safe (encoder may emit half)
    return v;
}

EncodedBeam VoxelFieldPredictor::encode_beam(const BeamParameters& beam) const {
    EncodedBeam e;
    e.beam = beam;
    if (beam_encoder_) {                       // two-graph model: run the encoder once
        e.latent = beam_encoder_->run_field_raw(beam);
        e.is_encoded = true;
    }                                          // else single-graph: carry the beam through
    return e;
}

FieldPrediction VoxelFieldPredictor::predict_voxelwise(const float* positions_xyz, size_t count,
                                                       const EncodedBeam& beam) const {
    FieldPrediction out; out.n_bins = out_bins_;
    const int M = static_cast<int>(count);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> res = run_positions(*impl_, positions_xyz, M, beam);  // binds positions zero-copy
    out.flux.resize(count);
    out.spectrum.resize(count * out_bins_);
    collect(res, out_bins_, count, out.flux.data(), out.spectrum.data());
    out.inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    out.dims = {M, 1, 1};
    return out;
}

FieldPrediction VoxelFieldPredictor::predict_voxelwise(const std::vector<std::array<float, 3>>& positions,
                                                       const EncodedBeam& beam) const {
    // std::array<float,3> is contiguous, so the vector is a packed [M,3] float buffer.
    return predict_voxelwise(reinterpret_cast<const float*>(positions.data()), positions.size(), beam);
}

FieldPrediction VoxelFieldPredictor::predict_voxelwise_absolute(
        const std::vector<std::array<float, 3>>& positions_m, const EncodedBeam& beam) const {
    // Normalise absolute (metre) positions by the dataset field box, then run the normalised query.
    const std::array<float, 3>& fd = domain().field_dimensions_m;
    std::vector<std::array<float, 3>> normalized;
    normalized.reserve(positions_m.size());
    for (const auto& p : positions_m) {
        normalized.push_back({ fd[0] > 0.f ? p[0] / fd[0] : p[0],
                               fd[1] > 0.f ? p[1] / fd[1] : p[1],
                               fd[2] > 0.f ? p[2] / fd[2] : p[2] });
    }
    return predict_voxelwise(normalized, beam);
}

FieldPrediction VolumeFieldPredictor::predict_volume(const BeamParameters& beam, std::array<int, 3> dims,
                                                     int max_inner_batch) const {
    (void)max_inner_batch;  // field-wise: the whole volume is one Run(), no tiling
    FieldPrediction out; out.n_bins = out_bins_; out.dims = dims;
    const size_t N = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    out.flux.resize(N); out.spectrum.resize(N * out_bins_);
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> res = run_field(*impl_, beam);
    collect(res, out_bins_, N, out.flux.data(), out.spectrum.data());
    out.inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

void VoxelFieldPredictor::tile_into(const EncodedBeam& enc, std::array<int, 3> dims, int max_inner_batch,
                                    float* flux_dst, float* spec_dst) const {
    // Tile the grid; every chunk runs at the SAME row count `CH` (the final partial chunk is padded
    // by repeating its last point, the padding dropped on write) so TensorRT sees a single static
    // input shape and builds/caches exactly one engine. Predictions are written straight into
    // `flux_dst`/`spec_dst` at the chunk's voxel offset — no intermediate buffer.
    const int D = dims[0], H = dims[1], W = dims[2];
    const size_t N = static_cast<size_t>(D) * H * W;
    const size_t CH = std::min(static_cast<size_t>(std::max(1, max_inner_batch)), N);
    std::vector<std::array<float, 3>> pts; pts.reserve(CH);
    size_t done = 0;
    auto flush = [&]() {
        const size_t valid = pts.size();
        while (pts.size() < CH) pts.push_back(pts.back());
        std::vector<Ort::Value> res = run_positions(*impl_, reinterpret_cast<const float*>(pts.data()),
                                                    static_cast<int>(pts.size()), enc);
        collect(res, out_bins_, valid,
                flux_dst ? flux_dst + done : nullptr,
                spec_dst ? spec_dst + done * out_bins_ : nullptr);
        done += valid; pts.clear();
    };
    for (int i = 0; i < D; ++i)
    for (int j = 0; j < H; ++j)
    for (int k = 0; k < W; ++k) {
        pts.push_back({i / std::max(1.f, D - 1.f), j / std::max(1.f, H - 1.f), k / std::max(1.f, W - 1.f)});
        if (pts.size() == CH) flush();
    }
    if (!pts.empty()) flush();
}

FieldPrediction VoxelFieldPredictor::predict_volume(const BeamParameters& beam, std::array<int, 3> dims,
                                                    int max_inner_batch) const {
    FieldPrediction out; out.n_bins = out_bins_; out.dims = dims;
    const size_t N = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
    out.flux.resize(N); out.spectrum.resize(N * out_bins_);
    auto t0 = std::chrono::high_resolution_clock::now();
    tile_into(encode_beam(beam), dims, max_inner_batch, out.flux.data(), out.spectrum.data());
    out.inference_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return out;
}

FieldPrediction VoxelFieldPredictor::predict_visible_voxels(const EncodedBeam& beam,
                                                                    std::array<int, 3> dims,
                                                                    const std::array<float, 16>& P) const {
    // Only voxels whose normalised [0,1]^3 centre projects inside the clip volume of `P` (the
    // sub-volume the projection actually sees). CPU visibility test; see compute_visible_voxels()
    // in vk/vulkan_field.h for the GPU compute-shader equivalent.
    const int D = dims[0], H = dims[1], W = dims[2];
    std::vector<std::array<float, 3>> pts;
    for (int i = 0; i < D; ++i)
    for (int j = 0; j < H; ++j)
    for (int k = 0; k < W; ++k) {
        const float x = i / std::max(1.f, D - 1.f);
        const float y = j / std::max(1.f, H - 1.f);
        const float z = k / std::max(1.f, W - 1.f);
        // Column-major 4×4: clip = P · (x,y,z,1).
        const float cx = P[0]*x + P[4]*y + P[8]*z  + P[12];
        const float cy = P[1]*x + P[5]*y + P[9]*z  + P[13];
        const float cz = P[2]*x + P[6]*y + P[10]*z + P[14];
        const float cw = P[3]*x + P[7]*y + P[11]*z + P[15];
        if (cw <= 0.f) continue;
        const float nx = cx / cw, ny = cy / cw, nz = cz / cw;
        if (nx >= -1.f && nx <= 1.f && ny >= -1.f && ny <= 1.f && nz >= 0.f && nz <= 1.f)
            pts.push_back({x, y, z});
    }
    auto out = predict_voxelwise(pts, beam);
    out.dims = {static_cast<int>(pts.size()), 1, 1};
    return out;
}

void VolumeFieldPredictor::predict_into_field(const BeamParameters& beam,
                                              DeviceCartesianRadiationField& out,
                                              int max_inner_batch) const {
    (void)max_inner_batch;  // field-wise: one Run() emits the whole volume
    const size_t n = out.voxel_count();
    auto channel = ensure_pred_layers(out, out_bins_, out_fp16_);
    float* spec_dst = channel->get_layer<float>(kSpectrumLayer);
    std::vector<Ort::Value> res = run_field(*impl_, beam);
    if (out_fp16_) {
        // Collect flux as float, then store fp16 (ONNX fp16 -> float -> fp16 is lossless).
        std::vector<float> flux(n);
        collect(res, out_bins_, n, flux.data(), spec_dst);
        auto* flux16 = channel->get_layer<RadFiled3D::Typing::float16>(kFluxLayer);
        for (size_t i = 0; i < n; ++i) flux16[i] = RadFiled3D::Typing::float16(flux[i]);
    } else {
        collect(res, out_bins_, n, channel->get_layer<float>(kFluxLayer), spec_dst);
    }
}

// ── Device-resident inference (zero-copy path; see field_predictors.h / cuda_vulkan_export.h) ─────────
struct DeviceFieldOutputs {
    std::vector<Ort::Value> values;   // volume path: the bound outputs own the CUDA device buffers
    const void* flux     = nullptr;   // device ptr: N scalars
    const void* spectrum = nullptr;   // device ptr: N*bins, or null
    size_t      n    = 0;
    bool        fp16 = false;
    void*       owned_buffer = nullptr;   // voxel path: a cudaMalloc'd buffer we own (the assembled flux)
    int         owned_device = -1;
};

void release_device_outputs(DeviceFieldOutputs* o) {
#if RFNN_CUDA_VULKAN_INTEROP
    if (o && o->owned_buffer) rfnn::cuda_vk::device_free(o->owned_device, o->owned_buffer);
#endif
    delete o;
}
const void* device_outputs_flux(const DeviceFieldOutputs* o)        { return o ? o->flux : nullptr; }
const void* device_outputs_spectrum(const DeviceFieldOutputs* o)    { return o ? o->spectrum : nullptr; }
size_t      device_outputs_voxel_count(const DeviceFieldOutputs* o) { return o ? o->n : 0; }
bool        device_outputs_is_fp16(const DeviceFieldOutputs* o)     { return o ? o->fp16 : false; }

DeviceFieldOutputs* VolumeFieldPredictor::predict_to_device(const BeamParameters& beam,
                                                            std::array<int, 3> dims, int device_id) const {
    // Beam-parameter inputs on CPU (the CUDA EP copies them to device). A per-voxel graph (position input)
    // can't run field-wise, so there is no whole-field device path here — caller falls back to the host path.
    const float dist = source_distance(beam);
    std::vector<std::vector<float>> buffers;
    std::vector<Ort::Value> inputs;
    for (const auto& n : impl_->in_names) {
        if (is_position_input(n)) return nullptr;
        make_beam_input(n, /*rows=*/1, beam, dist, impl_->param_ranges, impl_->mem, buffers, inputs);
    }

    try {
        // Bind every output to CUDA device memory: ORT allocates + writes the result on the GPU (no host
        // copy). The bound Ort::Values (kept in DeviceFieldOutputs::values) own those device buffers.
        Ort::MemoryInfo cuda_mem("Cuda", OrtDeviceAllocator, device_id, OrtMemTypeDefault);
        Ort::IoBinding binding(*impl_->session);
        for (size_t i = 0; i < impl_->in_names.size(); ++i)
            binding.BindInput(impl_->in_names[i].c_str(), inputs[i]);
        for (const auto& on : impl_->out_names)
            binding.BindOutput(on.c_str(), cuda_mem);
        impl_->session->Run(Ort::RunOptions{nullptr}, binding);

        std::vector<Ort::Value> outs = binding.GetOutputValues();
        auto out = std::make_unique<DeviceFieldOutputs>();
        out->n    = static_cast<size_t>(dims[0]) * dims[1] * dims[2];
        out->fp16 = out_fp16_;
        bool got_flux = false;
        for (auto& o : outs) {                                  // flux vs spectrum by trailing dim
            const auto shp = o.GetTensorTypeAndShapeInfo().GetShape();
            const bool is_spec = !shp.empty() && shp.back() == out_bins_;
            if (is_spec)          out->spectrum = o.GetTensorMutableRawData();
            else if (!got_flux) { out->flux     = o.GetTensorMutableRawData(); got_flux = true; }
        }
        out->values = std::move(outs);   // keep the device buffers alive past return
        if (!out->flux) return nullptr;
        return out.release();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[radfield3dnn] predict_to_device failed: %s\n", e.what());
        return nullptr;
    }
}

void VoxelFieldPredictor::predict_into_field(const BeamParameters& beam,
                                             DeviceCartesianRadiationField& out,
                                             int max_inner_batch) const {
    const glm::uvec3 c = out.get_voxel_counts();
    const size_t n = out.voxel_count();
    auto channel = ensure_pred_layers(out, out_bins_, out_fp16_);
    float* spec_dst = channel->get_layer<float>(kSpectrumLayer);
    const std::array<int, 3> dims{static_cast<int>(c.x), static_cast<int>(c.y), static_cast<int>(c.z)};
    // Encode once, then tile per-voxel predictions into the field's layer buffers.
    if (out_fp16_) {
        std::vector<float> flux(n);
        tile_into(encode_beam(beam), dims, max_inner_batch, flux.data(), spec_dst);
        auto* flux16 = channel->get_layer<RadFiled3D::Typing::float16>(kFluxLayer);
        for (size_t i = 0; i < n; ++i) flux16[i] = RadFiled3D::Typing::float16(flux[i]);
    } else {
        tile_into(encode_beam(beam), dims, max_inner_batch, channel->get_layer<float>(kFluxLayer), spec_dst);
    }
}

// Cap for the single-Run device path: above this voxel count the broadcast latent + activations would be a
// large device/host allocation, so fall back to chunked tiling. 1<<20 keeps the [N,L] latent host buffer
// under ~1 GB for typical latent widths; common field sizes (48³,64³,96³) stay single-Run.
static constexpr size_t kDeviceSingleRunMaxVoxels = 1u << 20;

DeviceFieldOutputs* VoxelFieldPredictor::predict_to_device(const BeamParameters& beam,
                                                           std::array<int, 3> dims, int device_id) const {
#if RFNN_CUDA_VULKAN_INTEROP
    // A voxel model queried over a volume yields the SAME N-value flux output as a volume model. The fast path
    // runs ALL N voxels in ONE Run with the outputs bound to CUDA device memory (IoBinding) — no host
    // download, no per-chunk IoBinding/padding, and (unlike a tiled assembly) the bound Ort::Value's device
    // buffer IS the result, so it's returned directly with no extra device_malloc/d2d copy. The bound values
    // are kept alive in DeviceFieldOutputs::values. Very large volumes fall back to the chunked path below.
    const bool prof = std::getenv("RFNN_PROFILE") != nullptr;
    const int D = dims[0], H = dims[1], W = dims[2];
    const size_t N = static_cast<size_t>(D) * H * W;
    if (N == 0) return nullptr;
    const auto te0 = std::chrono::high_resolution_clock::now();
    const EncodedBeam enc = encode_beam(beam);
    if (prof) std::fprintf(stderr, "[prof] encode_beam=%.3f ms\n",
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - te0).count());

    if (N <= kDeviceSingleRunMaxVoxels) {
        std::vector<std::array<float, 3>> pts(N);
        size_t idx = 0;
        for (int i = 0; i < D; ++i)
        for (int j = 0; j < H; ++j)
        for (int k = 0; k < W; ++k)
            pts[idx++] = { i / std::max(1.f, D - 1.f), j / std::max(1.f, H - 1.f), k / std::max(1.f, W - 1.f) };
        try {
            std::vector<Ort::Value> res = run_positions(*impl_, reinterpret_cast<const float*>(pts.data()),
                                                        static_cast<int>(N), enc, device_id);
            auto out = std::make_unique<DeviceFieldOutputs>();
            out->n = N; out->fp16 = out_fp16_;
            bool got_flux = false;
            for (auto& o : res) {
                const auto shp = o.GetTensorTypeAndShapeInfo().GetShape();
                const bool is_spec = !shp.empty() && shp.back() == out_bins_;
                if (is_spec)          out->spectrum = o.GetTensorMutableRawData();
                else if (!got_flux) { out->flux     = o.GetTensorMutableRawData(); got_flux = true; }
            }
            out->values = std::move(res);   // keep the device buffers alive past return
            if (!out->flux) return nullptr;
            return out.release();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[radfield3dnn] voxel predict_to_device (single-run) failed: %s\n", e.what());
            return nullptr;
        }
    }

    // Fallback for very large volumes: tile per-voxel queries (each chunk's ONNX output bound to CUDA device
    // memory) into ONE device flux buffer at its voxel offset. Bounded device/host working set per chunk.
    const size_t elem = out_fp16_ ? 2u : 4u;
    void* bigbuf = rfnn::cuda_vk::device_malloc(device_id, N * elem);
    if (!bigbuf) return nullptr;
    const size_t CH = std::min(kDeviceSingleRunMaxVoxels, N);
    std::vector<std::array<float, 3>> pts; pts.reserve(CH);
    size_t done = 0; bool ok = true;
    auto flush = [&]() {
        if (!ok) { pts.clear(); return; }
        const size_t valid = pts.size();
        while (pts.size() < CH) pts.push_back(pts.back());   // pad to CH (one static shape for TRT); drop padding
        std::vector<Ort::Value> res = run_positions(*impl_, reinterpret_cast<const float*>(pts.data()),
                                                     static_cast<int>(pts.size()), enc, device_id);
        const void* flux_dev = nullptr;
        for (auto& o : res) {
            const auto shp = o.GetTensorTypeAndShapeInfo().GetShape();
            const bool is_spec = !shp.empty() && shp.back() == out_bins_;
            if (!is_spec) { flux_dev = o.GetTensorMutableRawData(); break; }   // device ptr: this chunk's flux
        }
        if (!flux_dev ||
            !rfnn::cuda_vk::device_copy_d2d(static_cast<char*>(bigbuf) + done * elem, flux_dev, valid * elem)) {
            ok = false;
        }
        done += valid; pts.clear();
    };
    for (int i = 0; i < D; ++i)
    for (int j = 0; j < H; ++j)
    for (int k = 0; k < W; ++k) {
        pts.push_back({ i / std::max(1.f, D - 1.f), j / std::max(1.f, H - 1.f), k / std::max(1.f, W - 1.f) });
        if (pts.size() == CH) flush();
    }
    if (!pts.empty()) flush();

    if (!ok) { rfnn::cuda_vk::device_free(device_id, bigbuf); return nullptr; }
    auto out = std::make_unique<DeviceFieldOutputs>();
    out->flux = bigbuf; out->owned_buffer = bigbuf; out->owned_device = device_id;
    out->n = N; out->fp16 = out_fp16_;
    return out.release();
#else
    (void)beam; (void)dims; (void)device_id;
    return nullptr;
#endif
}

// ── VoxelFieldPredictor ────────────────────────────────────────────────────────────────────────
VoxelFieldPredictor::VoxelFieldPredictor(const void* trunk_bytes, size_t n,
                                         std::shared_ptr<VolumeFieldPredictor> beam_encoder,
                                         bool use_cuda)
    : VolumeFieldPredictor(trunk_bytes, n, use_cuda), beam_encoder_(std::move(beam_encoder)) {}

VoxelFieldPredictor::VoxelFieldPredictor(const void* trunk_bytes, size_t n,
                                         std::shared_ptr<VolumeFieldPredictor> beam_encoder,
                                         const ExecutionOptions& exec)
    : VolumeFieldPredictor(trunk_bytes, n, exec), beam_encoder_(std::move(beam_encoder)) {}

VoxelFieldPredictor::VoxelFieldPredictor(VolumeFieldPredictor&& trunk,
                                         std::shared_ptr<VolumeFieldPredictor> beam_encoder)
    : VolumeFieldPredictor(std::move(trunk)), beam_encoder_(std::move(beam_encoder)) {}

void VoxelFieldPredictor::predict_into_field(const BeamParameters& beam,
                                             DeviceCartesianRadiationField& out,
                                             const std::vector<std::array<int, 3>>& voxel_locations,
                                             int max_inner_batch) const {
    (void)max_inner_batch;  // one Run() over the (small) requested set
    const glm::uvec3 c = out.get_voxel_counts();
    const int H = static_cast<int>(c.y), W = static_cast<int>(c.z), D = static_cast<int>(c.x);
    const size_t n = out.voxel_count();
    const int bins = out_bins_;

    auto channel = ensure_pred_layers(out, bins, out_fp16_);
    float* spec_dst = channel->get_layer<float>(kSpectrumLayer);

    // Unpredicted voxels are -inf (flux and every spectrum bin); predicted ones get their values.
    const float neg_inf = -std::numeric_limits<float>::infinity();
    std::fill(spec_dst, spec_dst + n * static_cast<size_t>(bins), neg_inf);

    // Flux is scattered into a float work buffer, then stored as fp16 or float32 per the model.
    std::vector<float> flux_storage;
    float* flux_work = nullptr;
    if (out_fp16_) { flux_storage.assign(n, neg_inf); flux_work = flux_storage.data(); }
    else { flux_work = channel->get_layer<float>(kFluxLayer); std::fill(flux_work, flux_work + n, neg_inf); }

    auto store_flux16 = [&]() {
        if (!out_fp16_) return;
        auto* flux16 = channel->get_layer<RadFiled3D::Typing::float16>(kFluxLayer);
        for (size_t i = 0; i < n; ++i) flux16[i] = RadFiled3D::Typing::float16(flux_work[i]);
    };

    if (voxel_locations.empty()) { store_flux16(); return; }

    std::vector<std::array<float, 3>> pts;
    pts.reserve(voxel_locations.size());
    for (const auto& v : voxel_locations)
        pts.push_back({v[0] / std::max(1.f, D - 1.f), v[1] / std::max(1.f, H - 1.f), v[2] / std::max(1.f, W - 1.f)});

    const EncodedBeam enc = encode_beam(beam);
    const FieldPrediction pred = predict_voxelwise(pts, enc);

    // Scatter each prediction to its flat voxel index (((i*H)+j)*W+k, matching predict_volume).
    for (size_t m = 0; m < voxel_locations.size(); ++m) {
        const auto& v = voxel_locations[m];
        const size_t idx = (static_cast<size_t>(v[0]) * H + v[1]) * W + v[2];
        if (idx >= n) continue;
        if (m < pred.flux.size()) flux_work[idx] = pred.flux[m];
        for (int b = 0; b < bins; ++b) {
            const size_t s = m * static_cast<size_t>(bins) + b;
            if (s < pred.spectrum.size()) spec_dst[idx * static_cast<size_t>(bins) + b] = pred.spectrum[s];
        }
    }
    store_flux16();
}


// The RF3M parse+build (ModelStore::load[/_from_memory]) lives in model_io.cpp: it drives
// this file's predictor ctors / set_parameter_range through their public declarations, so the
// parse+build needs no ORT headers and the predictor it returns carries the package metadata.

// ════════════════════════════════════════════════════════════════════════════════════════════════
//  Zero-copy binding: the caller owns every buffer; the model reads and writes them in place.
//
//  Values are fed to the graph VERBATIM — a bound buffer must already hold what the network was
//  trained on. Use parameter_normalizer() to turn metric values (metres, degrees) into that space.
// ════════════════════════════════════════════════════════════════════════════════════════════════
namespace {

int bit_index(std::uint32_t flag) {
    int i = 0;
    while (flag > 1u) { flag >>= 1; ++i; }
    return i;
}

// Which interface flag a graph input carries. The exporters use varied names, so match by intent —
// the same rule make_beam_input() binds by.
ModelInput flag_for_graph_input(const std::string& n) {
    if (is_position_input(n))                          return ModelInput::Position;
    if (name_is(n, {"direction", "dir"}))              return ModelInput::BeamDirection;
    if (name_is(n, {"distance"}))                      return ModelInput::SourceDistance;
    if (name_is(n, {"origin", "src"}))                 return ModelInput::SourceOrigin3D;
    if (name_is(n, {"spectrum", "spec", "sp"}))        return ModelInput::TubeSpectrum;
    if (name_is(n, {"rect", "shape", "beam_shape"}))   return ModelInput::BeamCollimation;
    if (name_is(n, {"translation"}))                   return ModelInput::PatientTranslation3D;
    if (name_is(n, {"rotation"}))                      return ModelInput::PatientRotation3D;
    if (name_is(n, {"geometry"}))                      return ModelInput::GeometryMap;
    if (name_is(n, {"anode"}))                         return ModelInput::AnodeAngle;
    return ModelInput::None;
}

// The exporter auto-names the outputs ("div", "sigmoid") and their ORDER is not fixed, so neither
// carries meaning. The spectrum is whichever output's trailing dim is the bin count; the other is the
// flux. Same rule collect() applies to the results — here against the captured layout.
ModelOutput flag_for_output_index(const VolumeFieldPredictor::Impl& im, std::size_t i, int out_bins) {
    if (i >= im.out_shapes.size()) return ModelOutput::None;
    const auto& s = im.out_shapes[i];
    if (out_bins > 1 && s.size() >= 2 && s.back() == out_bins) return ModelOutput::Spectrum;
    return ModelOutput::Flux;
}

Ort::MemoryInfo memory_info_for(const MemoryRef& b) {
    if (b.device == DeviceKind::Cuda)
        return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, b.device_id, OrtMemTypeDefault);
    return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
}

// A tensor VIEWING caller memory — nothing is copied. `offset_elems` lets a tiled run point each
// chunk at its own slice of the caller's buffer.
Ort::Value view_tensor(const MemoryRef& b, std::vector<int64_t> shape, std::size_t offset_elems) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    auto mem = memory_info_for(b);
    if (b.dtype == ElementType::F16) {
        auto* p = static_cast<Ort::Float16_t*>(b.data) + offset_elems;
        return Ort::Value::CreateTensor<Ort::Float16_t>(mem, p, n, shape.data(), shape.size());
    }
    auto* p = static_cast<float*>(b.data) + offset_elems;
    return Ort::Value::CreateTensor<float>(mem, p, n, shape.data(), shape.size());
}

// Capture the real output shapes from a Run whose outputs ORT allocated. Called once per session:
// the rank and the trailing width do not change with the grid, only the row count does.
void capture_output_layout(VolumeFieldPredictor::Impl& im, std::vector<Ort::Value>& outs) {
    im.out_shapes.clear();
    for (auto& o : outs) im.out_shapes.push_back(o.GetTensorTypeAndShapeInfo().GetShape());
    im.out_layout_resolved = true;
}

}  // namespace

SessionMemory VolumeFieldPredictor::session_memory() const {
    SessionMemory s;
    s.provider  = impl_->provider + "ExecutionProvider";
    s.device    = uses_gpu() ? DeviceKind::Cuda : DeviceKind::Cpu;
    s.device_id = impl_->device_id;
    // Graph I/O stays fp32 even when the weights are fp16 (the exporter keeps I/O types), except for
    // models whose outputs were exported as fp16 — introspect() detects that.
    s.dtype = ElementType::F32;
    return s;
}

std::size_t VolumeFieldPredictor::required_elements(ModelInput flag) const {
    const std::size_t N = static_cast<std::size_t>(impl_->grid[0]) * impl_->grid[1] * impl_->grid[2];
    switch (flag) {
        case ModelInput::Position:             return N * 3;    // generated internally from the grid
        case ModelInput::QueryDirection:       return N * 3;
        case ModelInput::TubeSpectrum:         return static_cast<std::size_t>(input_spectrum_bins());
        case ModelInput::BeamDirection:        return 3;
        case ModelInput::SourceDistance:       return 1;
        case ModelInput::SourceOrigin3D:       return 3;
        case ModelInput::BeamCollimation:      return static_cast<std::size_t>(
                                                   rfnn::io::collimation_dims(domain_.collimation));
        case ModelInput::PatientTranslation3D: return 3;
        case ModelInput::PatientRotation3D:    return 3;
        case ModelInput::GeometryMap:          return N;
        case ModelInput::AnodeAngle:           return 1;
        default:                               return 0;
    }
}

std::size_t VolumeFieldPredictor::required_elements(ModelOutput flag) const {
    const std::size_t N = static_cast<std::size_t>(impl_->grid[0]) * impl_->grid[1] * impl_->grid[2];
    switch (flag) {
        case ModelOutput::Flux:     return N;
        case ModelOutput::Error:    return N;
        case ModelOutput::AirKerma: return N;
        case ModelOutput::Spectrum: return N * static_cast<std::size_t>(out_bins_);
        default:                    return 0;
    }
}

void VolumeFieldPredictor::set_voxel_grid(std::array<int, 3> voxel_counts) {
    for (int d : voxel_counts)
        if (d <= 0) throw BindingError("set_voxel_grid: every voxel count must be > 0, got {"
                                       + std::to_string(voxel_counts[0]) + ", "
                                       + std::to_string(voxel_counts[1]) + ", "
                                       + std::to_string(voxel_counts[2]) + "}");
    impl_->grid = voxel_counts;
    impl_->grid_set = true;
}

std::array<int, 3> VolumeFieldPredictor::voxel_grid() const { return impl_->grid; }

void VolumeFieldPredictor::clear_bindings() {
    for (auto& s : impl_->in_slots)  s = Impl::Slot{};
    for (auto& s : impl_->out_slots) s = Impl::Slot{};
    impl_->output_sinks.clear();
}

// Register a post-run sink (private; friended to rfnn::vk::VulkanImageTarget). Runs at the tail of
// every predict path once the outputs are written & synchronised. Cleared by clear_bindings().
void VolumeFieldPredictor::add_output_sink(std::function<void()> sink) {
    impl_->output_sinks.push_back(std::move(sink));
}

void VolumeFieldPredictor::bind_global_parameter(ModelInput flag, const MemoryRef& buffer,
                                                 bool convert_buffer) {
    if (flag == ModelInput::Position)
        throw BindingError("bind_global_parameter: Position is not a global parameter — it is the "
                           "per-query input, generated from set_voxel_grid().");
    if (!interface_.takes(flag))
        throw BindingError("bind_global_parameter: this model does not consume "
                           + to_string(flag) + ". It consumes: " + [&] {
                               std::string acc;
                               for (std::uint32_t b = 0; b < 32; ++b) {
                                   const auto f = static_cast<ModelInput>(1u << b);
                                   if (interface_.takes(f)) acc += (acc.empty() ? "" : ", ") + to_string(f);
                               }
                               return acc.empty() ? std::string("(nothing)") : acc;
                           }());

    const std::size_t need = required_elements(flag);
    const SessionMemory sm = session_memory();

    const bool fits    = buffer.bytes / element_size(buffer.dtype) >= need;
    const bool matches = buffer.device == sm.device && buffer.dtype == sm.dtype
                      && (buffer.device == DeviceKind::Cpu || buffer.device_id == sm.device_id);

    if (!fits || (!matches && !convert_buffer))
        throw BindingError(describe_binding_mismatch(to_string(flag), buffer, sm, need));

    auto& slot = impl_->in_slots[bit_index(static_cast<std::uint32_t>(flag))];
    slot.mem     = buffer;
    slot.convert = !matches;
    slot.bound   = true;
    slot.staging.clear();

    if (slot.convert) {
        // Snapshot NOW: staging decouples the model from the caller's buffer, so a later in-place
        // edit is not seen. (A directly-bound buffer IS re-read every run — that is the difference.)
        if (buffer.device != DeviceKind::Cpu)
            throw BindingError("bind_global_parameter(" + to_string(flag) + ", convert_buffer=true): "
                               "the buffer lives in " + to_string(buffer.device) + " memory, which "
                               "this runtime cannot read from the host to convert it. Bind a buffer "
                               "that matches the session (" + to_string(sm.device) + " / "
                               + to_string(sm.dtype) + "), or copy it to the host yourself first.");
        slot.staging.resize(need);
        if (buffer.dtype == ElementType::F16) {
            const auto* src = static_cast<const Ort::Float16_t*>(buffer.data);
            for (std::size_t i = 0; i < need; ++i) slot.staging[i] = static_cast<float>(src[i]);
        } else {
            std::memcpy(slot.staging.data(), buffer.data, need * sizeof(float));
        }
    }
}

void VolumeFieldPredictor::bind_output_layer(ModelOutput flag, const MemoryRef& buffer,
                                             bool convert_buffer) {
    if (!interface_.gives(flag))
        throw BindingError("bind_output_layer: this model does not produce " + to_string(flag) + ".");
    if (!impl_->grid_set)
        throw IncompleteSetupError("bind_output_layer(" + to_string(flag) + "): the output size "
                                   "depends on the grid — call set_voxel_grid() first.");

    const std::size_t need = required_elements(flag);
    SessionMemory sm = session_memory();
    sm.dtype = out_fp16_ ? ElementType::F16 : ElementType::F32;   // what the graph WRITES

    const bool fits    = buffer.bytes / element_size(buffer.dtype) >= need;
    const bool matches = buffer.device == sm.device && buffer.dtype == sm.dtype
                      && (buffer.device == DeviceKind::Cpu || buffer.device_id == sm.device_id);

    if (!fits || (!matches && !convert_buffer))
        throw BindingError(describe_binding_mismatch(to_string(flag), buffer, sm, need));

    auto& slot = impl_->out_slots[bit_index(static_cast<std::uint32_t>(flag))];
    slot.mem     = buffer;
    slot.convert = !matches;
    slot.bound   = true;
    slot.staging.clear();

    if (slot.convert) {
        if (buffer.device != DeviceKind::Cpu)
            throw BindingError("bind_output_layer(" + to_string(flag) + ", convert_buffer=true): the "
                               "buffer lives in " + to_string(buffer.device) + " memory, which this "
                               "runtime cannot write from the host. Bind a buffer that matches the "
                               "session (" + to_string(sm.device) + " / " + to_string(sm.dtype)
                               + ") — that path is zero-copy anyway.");
        slot.staging.resize(need);   // the model writes here; copied into the caller's buffer after run
    }
}

// Everything the caller still owes us, in ONE message — not one exception per missing piece.
void VolumeFieldPredictor::check_setup_complete() const {
    std::vector<std::string> missing;
    if (!impl_->grid_set)
        missing.push_back("the output grid — call set_voxel_grid({D, H, W})");

    for (std::uint32_t b = 0; b < 32; ++b) {
        const auto f = static_cast<ModelInput>(1u << b);
        if (f == ModelInput::Position || !interface_.takes(f)) continue;   // Position: generated here
        if (!impl_->in_slots[b].bound)
            missing.push_back("input " + to_string(f) + " — call bind_global_parameter(ModelInput::"
                              + to_string(f) + ", ...)");
    }

    bool any_output = false;
    for (const auto& s : impl_->out_slots) any_output |= s.bound;
    if (!any_output)
        missing.push_back("at least one output — call bind_output_layer(ModelOutput::Flux, ...)");

    if (missing.empty()) return;
    std::string m = "RadField3DNN: predict_volume() called before the model was fully set up. "
                    "Missing:\n";
    for (const auto& x : missing) m += "  * " + x + "\n";
    throw IncompleteSetupError(m);
}

// ── device-resident broadcast (GPU session only) ────────────────────────────────────────────────
// On a CUDA session the bound per-voxel path keeps the beam latent and the position grid in DEVICE
// memory across runs, so nothing crosses the PCIe bus per chunk. The primitives are ORT-only: the
// CUDA provider LIBRARY is registered with the Impl's env (which gives the env a device data-transfer
// and a shared device allocator), device tensors are allocated from that allocator, and CopyTensors
// moves data host→device / device→device. The classic per-session CUDA EP that runs the graphs does
// not expose an env-level data transfer on its own, hence the extra registration.

// One-time capability probe. Registers the CUDA EP library on the env and grabs the shared device
// allocator. Returns false (and latches device_broadcast_ok=0) on a CPU session or if any step is
// unavailable, so the caller falls back to the host broadcast path.
static bool ensure_device_broadcast(VolumeFieldPredictor::Impl& im) {
    if (im.device_broadcast_ok >= 0) return im.device_broadcast_ok == 1;
    im.device_broadcast_ok = 0;
    if (!im.on_device) return false;
    if (std::getenv("RFNN_FORCE_HOST_BROADCAST")) return false;   // benchmarking the host fallback

    // Registration is PROCESS-WIDE, keyed by this fixed name (a second name for the same library
    // crashes ORT), and it gives EVERY env a device data-transfer + shared allocator. A second
    // predictor's registration therefore throws "already registered" — harmless, the transfer still
    // works — so that throw is swallowed and only a missing shared allocator (no CUDA present) turns
    // the device path off. Bare soname: the loader resolves it next to libonnxruntime on the path.
    try {
        const char* soname = "libonnxruntime_providers_cuda.so";
        im.env.RegisterExecutionProviderLibrary(
            "rfnn_cuda_ep", std::basic_string<ORTCHAR_T>(soname, soname + std::char_traits<char>::length(soname)));
    } catch (const std::exception&) { /* already registered by an earlier predictor in this process */ }

    try {
        Ort::ThrowOnError(Ort::GetApi().GetSharedAllocator(im.env, im.cuda_mem, &im.shared_cuda_alloc));
        if (!im.shared_cuda_alloc) return false;
        im.device_broadcast_ok = 1;
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[radfield3dnn] device-resident broadcast unavailable (%s); using host path.\n",
                     e.what());
        im.shared_cuda_alloc = nullptr;
        return false;
    }
}

// A tensor VIEWING a slice of a device buffer: base device pointer + element offset, no copy.
// Pointer arithmetic on a device address is fine; only dereferencing it on the host is not.
static Ort::Value device_view(VolumeFieldPredictor::Impl& im, Ort::Value& base,
                              std::size_t offset_elems, std::vector<int64_t> shape) {
    std::size_t n = 1; for (auto d : shape) n *= static_cast<std::size_t>(d);
    float* p = base.GetTensorMutableData<float>() + offset_elems;   // device address
    return Ort::Value::CreateTensor<float>(im.cuda_mem, p, n, shape.data(), shape.size());
}

// CopyTensors one tensor src→dst through the env's device data transfer (H2D or D2D).
static void copy_tensor(VolumeFieldPredictor::Impl& im, const Ort::Value& src, Ort::Value& dst) {
    const OrtValue* s[1] = {static_cast<const OrtValue*>(src)};
    OrtValue*       d[1] = {static_cast<OrtValue*>(dst)};
    Ort::ThrowOnError(Ort::GetApi().CopyTensors(im.env, s, d, nullptr, 1));
}

// Ensure `dev_positions` holds the full [N,3] normalised grid, uploaded ONCE per grid. The fill order
// is the flat-index order ((i*H+j)*W+k), so row f of the buffer is the position of flat voxel f — the
// invariant predict_volume's chunks and predict_voxels' contiguous runs both slice against.
static void ensure_positions_device(VolumeFieldPredictor::Impl& im, int D, int H, int W) {
    if (im.dev_positions && im.dev_positions_grid == std::array<int,3>{D, H, W}) return;
    const std::size_t N = static_cast<std::size_t>(D) * H * W;
    std::vector<float> host(N * 3);
    std::size_t idx = 0;
    for (int i = 0; i < D; ++i)
    for (int j = 0; j < H; ++j)
    for (int k = 0; k < W; ++k) {
        host[idx++] = i / std::max(1.f, D - 1.f);
        host[idx++] = j / std::max(1.f, H - 1.f);
        host[idx++] = k / std::max(1.f, W - 1.f);
    }
    const int64_t shape[2] = {static_cast<int64_t>(N), 3};
    im.dev_positions = Ort::Value::CreateTensor(im.shared_cuda_alloc, shape, 2,
                                                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    Ort::Value hpos = Ort::Value::CreateTensor<float>(im.mem, host.data(), N * 3, shape, 2);
    copy_tensor(im, hpos, im.dev_positions);              // H2D, once per grid
    im.dev_positions_grid = {D, H, W};
}

// Ensure `dev_latent` is a [rows, L] device buffer (reused across runs). Reallocated only when the
// chunk row count or latent width changes.
static void ensure_latent_buffer(VolumeFieldPredictor::Impl& im, int rows, int L) {
    if (im.dev_latent && im.dev_latent_rows == rows && im.dev_latent_width == L) return;
    const int64_t shape[2] = {rows, L};
    im.dev_latent = Ort::Value::CreateTensor(im.shared_cuda_alloc, shape, 2,
                                             ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    im.dev_latent_rows = rows;
    im.dev_latent_width = L;
}

// Replicate the latent already sitting in dev_latent row 0 across `rows` rows, DEVICE-side, by
// geometric doubling (~log2(rows) D2D copies of growing spans) — the device analogue of the host
// broadcast trick. No data leaves the GPU.
static void broadcast_latent_device(VolumeFieldPredictor::Impl& im, int rows, int L) {
    for (int filled = 1; filled < rows; ) {
        const int copy = std::min(filled, rows - filled);
        const std::vector<int64_t> sh{copy, L};
        Ort::Value src = device_view(im, im.dev_latent, 0, sh);
        Ort::Value dst = device_view(im, im.dev_latent, static_cast<std::size_t>(filled) * L, sh);
        copy_tensor(im, src, dst);
        filled += copy;
    }
}

// Build the graph's input tensors from the bound slots. `rows` is the batch the graph runs at (the
// beam encoder / a field-wise trunk run at 1; a tiled trunk at its chunk size).
static void bind_inputs_from_slots(VolumeFieldPredictor::Impl& im, const ModelInterface& iface,
                                   const std::vector<std::string>& names, int rows,
                                   Ort::IoBinding& binding, std::vector<Ort::Value>& keepalive) {
    for (const auto& n : names) {
        const ModelInput flag = flag_for_graph_input(n);
        if (flag == ModelInput::None)
            throw BindingError("unmapped ONNX input '" + n + "' — this graph consumes a tensor the "
                               "binding API has no flag for.");
        const auto& slot = im.in_slots[bit_index(static_cast<std::uint32_t>(flag))];
        if (!slot.bound)
            throw IncompleteSetupError("input " + to_string(flag) + " is not bound");

        const std::size_t w = slot.convert ? slot.staging.size() : 0;
        std::vector<int64_t> shape{rows, 0};

        if (slot.convert) {
            // The staged copy lives on the host; ONNX Runtime moves it to the EP each Run().
            shape[1] = static_cast<int64_t>(w);
            keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                im.mem, const_cast<float*>(slot.staging.data()), slot.staging.size(),
                shape.data(), static_cast<size_t>(shape.size())));
        } else {
            // Zero-copy: the graph reads the caller's memory directly, so writing new values into it
            // in place is picked up by the next run with no rebinding.
            const std::size_t elems = slot.mem.bytes / element_size(slot.mem.dtype);
            shape[1] = static_cast<int64_t>(elems);
            keepalive.emplace_back(view_tensor(slot.mem, shape, 0));
        }
        binding.BindInput(n.c_str(), keepalive.back());
    }
}

// Bind the caller's output memory for a run covering `rows` voxels starting at `offset` of the grid.
// Each chunk points ORT at its own slice, so a tiled model writes straight into the final buffer.
static void bind_outputs_from_slots(VolumeFieldPredictor::Impl& im, const ModelInterface& iface,
                                    int out_bins, int rows, std::size_t offset,
                                    Ort::IoBinding& binding, std::vector<Ort::Value>& keepalive) {
    for (std::size_t i = 0; i < im.out_names.size(); ++i) {
        const ModelOutput flag = flag_for_output_index(im, i, out_bins);
        const int b = (flag == ModelOutput::None) ? -1 : bit_index(static_cast<std::uint32_t>(flag));
        auto* slot = (b >= 0) ? &im.out_slots[b] : nullptr;

        if (!slot || !slot->bound) {
            // Not asked for, but the graph still produces it. Bind it to the SESSION's device so ORT
            // scratch-allocates it there and drops it — on a CUDA session binding CPU memory here would
            // download the whole tensor (e.g. an [N,bins] spectrum) across the bus every run.
            binding.BindOutput(im.out_names[i].c_str(), im.on_device ? im.cuda_mem : im.mem);
            continue;
        }
        // Shape it exactly as the graph produced it in the probe run: the flux head is rank-1
        // ([batch]) on these exports while the spectrum is [batch, bins]. ORT rejects a mismatch.
        const auto& decl = im.out_shapes[i];
        const int width = (decl.size() >= 2 && decl.back() > 0) ? static_cast<int>(decl.back()) : 1;
        std::vector<int64_t> shape = (decl.size() >= 2) ? std::vector<int64_t>{rows, width}
                                                        : std::vector<int64_t>{rows};
        if (slot->convert) {
            keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                im.mem, slot->staging.data() + offset * width,
                static_cast<std::size_t>(rows) * width, shape.data(), shape.size()));
        } else {
            keepalive.emplace_back(view_tensor(slot->mem, shape, offset * width));
        }
        binding.BindOutput(im.out_names[i].c_str(), keepalive.back());
    }
}

// convert_buffer outputs: the model wrote into staging; move it into the caller's buffer.
static void flush_converted_outputs(VolumeFieldPredictor::Impl& im) {
    for (auto& slot : im.out_slots) {
        if (!slot.bound || !slot.convert) continue;
        if (slot.mem.dtype == ElementType::F16) {
            auto* dst = static_cast<Ort::Float16_t*>(slot.mem.data);
            for (std::size_t i = 0; i < slot.staging.size(); ++i)
                dst[i] = Ort::Float16_t(slot.staging[i]);
        } else {
            std::memcpy(slot.mem.data, slot.staging.data(), slot.staging.size() * sizeof(float));
        }
    }
    // Single choke point at the tail of every predict path: relayout device outputs into their final
    // resource (e.g. a Vulkan image) now that the run's writes are synchronised. Empty by default.
    for (auto& sink : im.output_sinks)
        if (sink) sink();
}

// Field-wise model: one graph, one Run(), the whole volume.
void VolumeFieldPredictor::predict_volume() {
    check_setup_complete();

    Ort::IoBinding binding(*impl_->session);
    std::vector<Ort::Value> keepalive;
    keepalive.reserve(impl_->in_names.size() + impl_->out_names.size());

    const std::size_t N = static_cast<std::size_t>(impl_->grid[0]) * impl_->grid[1] * impl_->grid[2];

    if (!impl_->out_layout_resolved) {
        // One probe with ORT-allocated outputs to learn their real shapes (the exported graph does
        // not declare them). Everything after this binds the caller's memory directly.
        Ort::IoBinding probe(*impl_->session);
        std::vector<Ort::Value> pk;
        bind_inputs_from_slots(*impl_, interface_, impl_->in_names, 1, probe, pk);
        for (const auto& on : impl_->out_names) probe.BindOutput(on.c_str(), impl_->mem);
        impl_->session->Run(Ort::RunOptions{nullptr}, probe);
        auto outs = probe.GetOutputValues();
        capture_output_layout(*impl_, outs);
    }

    bind_inputs_from_slots(*impl_, interface_, impl_->in_names, 1, binding, keepalive);
    bind_outputs_from_slots(*impl_, interface_, out_bins_, static_cast<int>(N), 0, binding, keepalive);

    impl_->session->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
    flush_converted_outputs(*impl_);
}

// Per-voxel model: encode the beam from the bound globals, then tile the grid, each chunk writing
// straight into its own slice of the caller's output buffers. On a GPU session the fast path keeps
// the latent + positions device-resident; it falls back to the host tiling path below.
void VoxelFieldPredictor::predict_volume() {
    check_setup_complete();
    if (!beam_encoder_)
        throw IncompleteSetupError("this per-voxel package has no beam_encoder graph, so its bound "
                                   "globals cannot be encoded.");
    if (impl_->on_device && predict_volume_device()) return;

    const std::vector<float> latent = encode_bound_beam();
    const int L = static_cast<int>(latent.size());
    ensure_output_layout(latent);

    // Tile the grid. Chunks are sized like the allocating path; the final chunk runs at its true row
    // count (no padding) — padding would write past the end of the caller's buffer.
    const int D = impl_->grid[0], H = impl_->grid[1], W = impl_->grid[2];
    const std::size_t N = static_cast<std::size_t>(D) * H * W;
    const std::size_t CH = std::min<std::size_t>(65536, N);

    std::vector<std::array<float, 3>> pts;
    pts.reserve(CH);
    std::vector<float> bcast(CH * L);   // reused across chunks — the broadcast latent, host-side
    for (std::size_t r = 0; r < CH; ++r)
        std::memcpy(&bcast[r * L], latent.data(), L * sizeof(float));
    std::size_t done = 0;

    auto flush = [&]() {
        const int rows = static_cast<int>(pts.size());
        Ort::IoBinding binding(*impl_->session);
        std::vector<Ort::Value> keepalive;

        for (const auto& n : impl_->in_names) {
            if (is_position_input(n)) {
                const int64_t shape[2] = {rows, 3};
                keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                    impl_->mem, reinterpret_cast<float*>(pts.data()),
                    static_cast<std::size_t>(rows) * 3, shape, 2));
            } else {
                const int64_t shape[2] = {rows, L};
                keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                    impl_->mem, bcast.data(), static_cast<std::size_t>(rows) * L, shape, 2));
            }
            binding.BindInput(n.c_str(), keepalive.back());
        }
        bind_outputs_from_slots(*impl_, interface_, out_bins_, rows, done, binding, keepalive);
        impl_->session->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();   // the writes go to the CALLER's memory: wait for them
        done += static_cast<std::size_t>(rows);
        pts.clear();
    };

    for (int i = 0; i < D; ++i)
    for (int j = 0; j < H; ++j)
    for (int k = 0; k < W; ++k) {
        pts.push_back({i / std::max(1.f, D - 1.f), j / std::max(1.f, H - 1.f), k / std::max(1.f, W - 1.f)});
        if (pts.size() == CH) flush();
    }
    if (!pts.empty()) flush();

    flush_converted_outputs(*impl_);
}

// Run the beam encoder over the bound globals, leave the latent in DEVICE memory (encoder output
// bound to the session's device), copy it into row 0 of the persistent device buffer, and broadcast
// it device-side to `chunk_rows` rows. Re-run every predict, so an in-place edit of a bound input is
// seen. The latent width is learned from the encoder output and cached in dev_latent_width.
void VoxelFieldPredictor::encode_broadcast_device(int chunk_rows) {
    auto& enc_impl = *beam_encoder_->impl_;
    Ort::IoBinding ebind(*enc_impl.session);
    std::vector<Ort::Value> ekeep;
    bind_inputs_from_slots(*impl_, interface_, enc_impl.in_names, 1, ebind, ekeep);
    for (const auto& on : enc_impl.out_names) ebind.BindOutput(on.c_str(), impl_->cuda_mem);
    enc_impl.session->Run(Ort::RunOptions{nullptr}, ebind);

    std::vector<Ort::Value> ev = ebind.GetOutputValues();
    const int L = static_cast<int>(ev[0].GetTensorTypeAndShapeInfo().GetShape().back());
    ensure_latent_buffer(*impl_, chunk_rows, L);
    Ort::Value dst0 = device_view(*impl_, impl_->dev_latent, 0, {1, L});
    copy_tensor(*impl_, ev[0], dst0);                    // D2D [1,L]: encoder device output -> row 0
    broadcast_latent_device(*impl_, chunk_rows, L);
}

// One tiled run of `rows` voxels from flat `offset`: position and latent inputs are device sub-views
// (dev_positions / dev_latent), outputs are the caller's device buffers bound at the run's offset.
void VoxelFieldPredictor::run_span_device(std::size_t offset, int rows, int L) {
    Ort::IoBinding binding(*impl_->session);
    std::vector<Ort::Value> keepalive;
    for (const auto& n : impl_->in_names) {
        if (is_position_input(n))
            keepalive.emplace_back(device_view(*impl_, impl_->dev_positions, offset * 3, {rows, 3}));
        else
            keepalive.emplace_back(device_view(*impl_, impl_->dev_latent, 0, {rows, L}));
        binding.BindInput(n.c_str(), keepalive.back());
    }
    bind_outputs_from_slots(*impl_, interface_, out_bins_, rows, offset, binding, keepalive);
    impl_->session->Run(Ort::RunOptions{nullptr}, binding);
    binding.SynchronizeOutputs();
}

bool VoxelFieldPredictor::predict_volume_device() {
    if (!ensure_device_broadcast(*impl_)) return false;
    const int D = impl_->grid[0], H = impl_->grid[1], W = impl_->grid[2];
    const std::size_t N = static_cast<std::size_t>(D) * H * W;
    const int CH = static_cast<int>(std::min<std::size_t>(65536, N));
    try {
        encode_broadcast_device(CH);
        const int L = impl_->dev_latent_width;
        if (!impl_->out_layout_resolved) ensure_output_layout(std::vector<float>(L, 0.f));
        ensure_positions_device(*impl_, D, H, W);
        for (std::size_t done = 0; done < N; done += CH)
            run_span_device(done, static_cast<int>(std::min<std::size_t>(CH, N - done)), L);
        flush_converted_outputs(*impl_);
        return true;
    } catch (const std::exception& e) {
        // A partial device run left some chunks written; the host fallback re-runs the whole grid, so
        // the final buffer is correct. Latch the fast path off so a genuine incapability is paid once.
        std::fprintf(stderr, "[radfield3dnn] device predict_volume failed (%s); host fallback.\n", e.what());
        impl_->device_broadcast_ok = 0;
        return false;
    }
}

// ── metric -> normalised, written into the bound buffer ──────────────────────────────────────────
ParameterNormalizer VolumeFieldPredictor::parameter_normalizer() const {
    return ParameterNormalizer(this);
}

std::vector<float> ParameterNormalizer::normalize(ModelInput flag, const float* values,
                                                  std::size_t count, Unit unit) const {
    const std::size_t need = pred_->required_elements(flag);
    if (count != need)
        throw BindingError("ParameterNormalizer: " + to_string(flag) + " takes "
                           + std::to_string(need) + " values, got " + std::to_string(count) + ".");

    // 1. the caller's unit -> the canonical unit the domain records (metres / degrees)
    auto to_canonical = [unit](float v) {
        switch (unit) {
            case Unit::Millimetres: return v * 1e-3f;                        // -> m
            case Unit::Radians:     return v * (180.0f / 3.14159265358979f); // -> deg
            case Unit::Metres:
            case Unit::Degrees:
            case Unit::Normalized:  return v;
        }
        return v;
    };

    // 2. canonical -> the network's space, using what the MODEL recorded about itself.
    const auto& fd = pred_->domain_.field_dimensions_m;
    const bool spatial = (flag == ModelInput::SourceOrigin3D);

    std::vector<float> out(count);
    for (std::size_t i = 0; i < count; ++i) {
        const float v = to_canonical(values[i]);
        if (unit == Unit::Normalized) {
            out[i] = v;                        // the caller says it is already in the network's space
        } else if (spatial) {
            // A location in METRES -> the field-relative [0,1]^3 the positions live in. The box is the
            // training dataset's, carried in the domain; without it a metric origin lands far outside.
            const float box = fd[i % 3];
            out[i] = (box > 0.f) ? v / box : v;
        } else {
            // Everything the model learned in normalised form (a distance in metres, an opening angle
            // in degrees) is clipped to the domain's [min,max] and mapped to [0,1]. Inputs with no
            // recorded range (a unit direction, the spectrum histogram) pass through unchanged.
            out[i] = normalize_metric(graph_name_for(flag), v, pred_->impl_->param_ranges);
        }
    }
    return out;
}

void ParameterNormalizer::write(ModelInput flag, const float* values, std::size_t count,
                                Unit unit) const {
    auto& slot = pred_->impl_->in_slots[bit_index(static_cast<std::uint32_t>(flag))];
    if (!slot.bound)
        throw IncompleteSetupError("ParameterNormalizer::write(" + to_string(flag) + "): nothing is "
                                   "bound for that input — bind_global_parameter() it first.");
    if (!slot.convert && slot.mem.device != DeviceKind::Cpu)
        throw BindingError(
            "ParameterNormalizer::write(" + to_string(flag) + "): the bound buffer is "
            + to_string(slot.mem.device) + " memory, and the host cannot write it. Two ways round it:\n"
            "  1. Bind a HOST buffer for this input with convert_buffer = true. The runtime then owns a\n"
            "     staged copy, write() fills that, and it is uploaded to the device on every run. Right\n"
            "     for the small metric inputs (a distance, an angle) this converter exists for.\n"
            "  2. Keep the device buffer and use normalize(), which RETURNS the values — upload them\n"
            "     with your own framework (torch: buf.copy_(torch.from_numpy(norm.normalize(...)))).");
    if (!slot.convert && slot.mem.dtype != ElementType::F32)
        throw BindingError("ParameterNormalizer::write(" + to_string(flag) + "): the bound buffer is "
                           + to_string(slot.mem.dtype) + "; the normalizer writes float32.");

    const std::vector<float> v = normalize(flag, values, count, unit);
    float* dst = slot.convert ? slot.staging.data() : static_cast<float*>(slot.mem.data);
    std::memcpy(dst, v.data(), v.size() * sizeof(float));
}

const std::string& ParameterNormalizer::graph_name_for(ModelInput flag) {
    static const std::map<ModelInput, std::string> kNames = {
        {ModelInput::BeamDirection,        "direction"},
        {ModelInput::SourceDistance,       "distance"},
        {ModelInput::SourceOrigin3D,       "origin"},
        {ModelInput::TubeSpectrum,         "spectrum"},
        {ModelInput::BeamCollimation,      "opening_angle"},
        {ModelInput::PatientTranslation3D, "translation"},
        {ModelInput::PatientRotation3D,    "rotation"},
        {ModelInput::AnodeAngle,           "anode_angle"},
    };
    static const std::string kNone;
    auto it = kNames.find(flag);
    return it == kNames.end() ? kNone : it->second;
}

std::vector<float> VoxelFieldPredictor::encode_bound_beam() const {
    auto& enc_impl = *beam_encoder_->impl_;
    Ort::IoBinding binding(*enc_impl.session);
    std::vector<Ort::Value> keep;
    bind_inputs_from_slots(*impl_, interface_, enc_impl.in_names, 1, binding, keep);
    for (const auto& on : enc_impl.out_names) binding.BindOutput(on.c_str(), enc_impl.mem);
    enc_impl.session->Run(Ort::RunOptions{nullptr}, binding);

    auto vals = enc_impl.session ? binding.GetOutputValues() : std::vector<Ort::Value>{};
    const float* lat = vals[0].GetTensorData<float>();
    const auto shape = vals[0].GetTensorTypeAndShapeInfo().GetShape();
    const int L = static_cast<int>(shape.back());
    return std::vector<float>(lat, lat + L);
}

void VoxelFieldPredictor::ensure_output_layout(const std::vector<float>& latent) const {
    if (impl_->out_layout_resolved) return;
    // The exported trunk declares no output shapes (ORT reports rank 0), so run one row with
    // ORT-allocated outputs and record what actually comes back. Every bind after this is exact.
    Ort::IoBinding probe(*impl_->session);
    std::vector<Ort::Value> keep;
    const std::array<float, 3> p0{0.f, 0.f, 0.f};
    std::vector<float> lat1(latent);
    const int L = static_cast<int>(lat1.size());
    for (const auto& n : impl_->in_names) {
        if (is_position_input(n)) {
            const int64_t sh[2] = {1, 3};
            keep.emplace_back(Ort::Value::CreateTensor<float>(
                impl_->mem, const_cast<float*>(p0.data()), 3, sh, 2));
        } else {
            const int64_t sh[2] = {1, L};
            keep.emplace_back(Ort::Value::CreateTensor<float>(
                impl_->mem, lat1.data(), lat1.size(), sh, 2));
        }
        probe.BindInput(n.c_str(), keep.back());
    }
    for (const auto& on : impl_->out_names) probe.BindOutput(on.c_str(), impl_->mem);
    impl_->session->Run(Ort::RunOptions{nullptr}, probe);
    auto outs = probe.GetOutputValues();
    capture_output_layout(*impl_, outs);
}

void VoxelFieldPredictor::predict_voxels(const std::vector<std::array<int, 3>>& voxel_indices) {
    check_setup_complete();
    if (!beam_encoder_)
        throw IncompleteSetupError("this per-voxel package has no beam_encoder graph, so its bound "
                                   "globals cannot be encoded.");
    if (voxel_indices.empty()) return;

    const int D = impl_->grid[0], H = impl_->grid[1], W = impl_->grid[2];

    // Flat offsets into the bound grid, sorted so that neighbours land next to each other. Writing a
    // voxel means writing exactly its offset — everything not listed keeps whatever the caller had.
    std::vector<std::size_t> flat;
    flat.reserve(voxel_indices.size());
    for (const auto& v : voxel_indices) {
        if (v[0] < 0 || v[0] >= D || v[1] < 0 || v[1] >= H || v[2] < 0 || v[2] >= W)
            throw BindingError("predict_voxels: voxel (" + std::to_string(v[0]) + ", "
                               + std::to_string(v[1]) + ", " + std::to_string(v[2])
                               + ") lies outside the bound grid " + std::to_string(D) + "x"
                               + std::to_string(H) + "x" + std::to_string(W) + ".");
        flat.push_back((static_cast<std::size_t>(v[0]) * H + v[1]) * W + v[2]);
    }
    std::sort(flat.begin(), flat.end());
    flat.erase(std::unique(flat.begin(), flat.end()), flat.end());

    if (impl_->on_device && predict_voxels_device(flat)) return;

    const std::vector<float> latent = encode_bound_beam();
    const int L = static_cast<int>(latent.size());
    ensure_output_layout(latent);

    // Group into maximal contiguous runs: one Run() per run, its outputs bound AT the run's offset,
    // so ORT writes the caller's memory in place with no scatter and no staging.
    std::size_t r = 0;
    while (r < flat.size()) {
        std::size_t e = r + 1;
        while (e < flat.size() && flat[e] == flat[e - 1] + 1) ++e;
        const int rows = static_cast<int>(e - r);
        const std::size_t offset = flat[r];

        std::vector<std::array<float, 3>> pts;
        pts.reserve(rows);
        for (std::size_t f = flat[r]; f <= flat[e - 1]; ++f) {
            const int k = static_cast<int>(f % W);
            const int j = static_cast<int>((f / W) % H);
            const int i = static_cast<int>(f / (static_cast<std::size_t>(W) * H));
            pts.push_back({i / std::max(1.f, D - 1.f),
                           j / std::max(1.f, H - 1.f),
                           k / std::max(1.f, W - 1.f)});
        }

        std::vector<float> bcast(static_cast<std::size_t>(rows) * L);
        for (int q = 0; q < rows; ++q)
            std::memcpy(&bcast[static_cast<std::size_t>(q) * L], latent.data(), L * sizeof(float));

        Ort::IoBinding binding(*impl_->session);
        std::vector<Ort::Value> keepalive;
        for (const auto& n : impl_->in_names) {
            if (is_position_input(n)) {
                const int64_t shape[2] = {rows, 3};
                keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                    impl_->mem, reinterpret_cast<float*>(pts.data()),
                    static_cast<std::size_t>(rows) * 3, shape, 2));
            } else {
                const int64_t shape[2] = {rows, L};
                keepalive.emplace_back(Ort::Value::CreateTensor<float>(
                    impl_->mem, bcast.data(), bcast.size(), shape, 2));
            }
            binding.BindInput(n.c_str(), keepalive.back());
        }
        bind_outputs_from_slots(*impl_, interface_, out_bins_, rows, offset, binding, keepalive);
        impl_->session->Run(Ort::RunOptions{nullptr}, binding);
        binding.SynchronizeOutputs();
        r = e;
    }

    flush_converted_outputs(*impl_);
}

// GPU-session predict_voxels: same contiguous-run grouping, but positions come from the persistent
// device grid (row f == flat voxel f) and the latent from the device broadcast, so no PCIe traffic
// per run. Runs longer than the broadcast buffer are sub-tiled. Returns false (host fallback) when
// the device data-transfer is unavailable.
bool VoxelFieldPredictor::predict_voxels_device(const std::vector<std::size_t>& flat) {
    if (!ensure_device_broadcast(*impl_)) return false;
    const int D = impl_->grid[0], H = impl_->grid[1], W = impl_->grid[2];
    const std::size_t N = static_cast<std::size_t>(D) * H * W;
    const int CH = static_cast<int>(std::min<std::size_t>(65536, N));
    try {
        encode_broadcast_device(CH);
        const int L = impl_->dev_latent_width;
        if (!impl_->out_layout_resolved) ensure_output_layout(std::vector<float>(L, 0.f));
        ensure_positions_device(*impl_, D, H, W);

        std::size_t r = 0;
        while (r < flat.size()) {
            std::size_t e = r + 1;
            while (e < flat.size() && flat[e] == flat[e - 1] + 1) ++e;
            // A contiguous flat range maps to contiguous rows of dev_positions; sub-tile at CH so a
            // run never exceeds the broadcast buffer.
            for (std::size_t off = flat[r]; off < flat[e - 1] + 1; off += CH)
                run_span_device(off, static_cast<int>(std::min<std::size_t>(CH, flat[e - 1] + 1 - off)), L);
            r = e;
        }
        flush_converted_outputs(*impl_);
        return true;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[radfield3dnn] device predict_voxels failed (%s); host fallback.\n", ex.what());
        impl_->device_broadcast_ok = 0;
        return false;
    }
}

}  // namespace radfield3dnn
