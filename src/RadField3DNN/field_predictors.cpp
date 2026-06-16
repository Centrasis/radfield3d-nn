#include "radfield3d-nn/field_predictors.h"

#include "radfield3d-nn/device_radiation_field.h"
#include "radfield3d-nn/model_io.h"

#include <onnxruntime_cxx_api.h>

#include <RadFiled3D/Voxel.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
    std::vector<std::vector<int64_t>> in_shapes;  // graph-declared shapes (dyn dims < 0)
    // Metric [min,max] ranges per beam-parameter name (from the RF3M ModelDomain). Used to
    // clip+normalise the inputs the model trained on in normalised form (e.g. "distance").
    std::map<std::string, std::array<float, 2>> param_ranges;
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
    try {
        im.opts.AppendExecutionProvider_TensorRT_V2(*trt);  // throws if EP not registrable
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
    if (!exec.use_gpu) return;  // CPU only

    if (exec.use_tensorrt) append_tensorrt(im, exec);

    // CUDA EP: claims any op TRT left behind, and is the GPU path when TRT is off/absent.
    try { OrtCUDAProviderOptions cuda{}; cuda.device_id = exec.device_id; im.opts.AppendExecutionProvider_CUDA(cuda); }
    catch (const std::exception&) { /* no CUDA EP -> CPU fallback */ }
}

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
                                                std::vector<std::string> graph_names) {
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
                                      const EncodedBeam& beam) {
    std::vector<std::vector<float>> buffers;  // own the (beam/latent) data until Run() returns
    std::vector<Ort::Value> inputs;
    buffers.reserve(im.in_names.size());

    for (const auto& n : im.in_names) {
        if (is_position_input(n)) {
            const int64_t shape[2] = {rows, 3};
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                im.mem, const_cast<float*>(positions_xyz), static_cast<size_t>(rows) * 3, shape, 2));
        } else if (beam.is_encoded) {
            // The trunk's non-position input is the latent: broadcast it over `rows`.
            const int L = static_cast<int>(beam.latent.size());
            std::vector<float> d(static_cast<size_t>(rows) * L);
            for (int i = 0; i < rows; ++i)
                std::memcpy(&d[static_cast<size_t>(i) * L], beam.latent.data(), L * sizeof(float));
            buffers.emplace_back(std::move(d));
            const int64_t shape[2] = {rows, L};
            inputs.emplace_back(Ort::Value::CreateTensor<float>(
                im.mem, buffers.back().data(), buffers.back().size(), shape, 2));
        } else {
            make_beam_input(n, rows, beam.beam, source_distance(beam.beam), im.param_ranges,
                            im.mem, buffers, inputs);
        }
    }
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

}  // namespace radfield3dnn

// The RF3M parse+build (ModelStore::load[/_from_memory]) lives in model_io.cpp: it drives
// this file's predictor ctors / set_parameter_range through their public declarations, so the
// parse+build needs no ORT headers and the predictor it returns carries the package metadata.
