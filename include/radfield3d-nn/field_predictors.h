#pragma once
// Field predictors — deployment-time inference of an EXPORTED RadField3D-NN model through ONNX
// Runtime, with no dependency on tiny-cuda-nn or libtorch. A model trained by the Python side is
// exported to ONNX (lowered to standard ops) and run here in a bare C++ runtime.
//
// Two predictor types, distinguished by type() (PredictorType):
//
//   * VolumeFieldPredictor — a field-wise model (CNN / whole-volume). Runs ONE "trunk" graph that
//                            maps beam parameters -> the full D×H×W volume in a single Run().
//
//   * VoxelFieldPredictor  — a per-voxel implicit model (MLP / NeRF / SIREN). Derives from
//                            VolumeFieldPredictor (it can also assemble a whole volume, by tiling
//                            per-voxel queries) and adds predict_voxelwise() + a beam-encoder
//                            (beam -> latent) trunk pair. The factory builds the right type.
//
// Output layout matches the trained models: flux is the joined per-voxel relative flux; spectrum
// is the per-voxel histogram (n_bins, default 32).

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <radfield3d-nn/model_domain.h>      // rfnn::io::{ModelDomain,ModelProvenance} carried by a loaded predictor
#include <radfield3d-nn/model_interface.h>   // radfield3dnn::ModelInterface — the declared I/O of a loaded package
#include <radfield3d-nn/model_binding.h>     // MemoryRef / SessionMemory — caller-owned zero-copy buffers

namespace Ort { struct Env; struct Session; struct MemoryInfo; }  // fwd-decl (onnxruntime_cxx_api.h in .cpp)

namespace rfnn { namespace vk { class VulkanImageTarget; } }  // fwd-decl: registers a post-run output sink

namespace radfield3dnn {

class DeviceCartesianRadiationField;  // device_radiation_field.h (fwd-decl; only needed by predict_into_field)

// Beam geometry / spectrum that conditions every prediction. Units match the
// dataset's DirectionalInput: origin is field-relative ([0,1], field centre at 0.5),
// direction is a unit vector, rect is the collimation size in metres at the isocentre.
struct BeamParameters {
    std::array<float, 3> direction{0.f, 0.f, -1.f};
    std::array<float, 3> origin{0.5f, 0.5f, 0.5f};
    std::vector<float>   spectrum;                 // tube spectrum histogram (raw bins)
    std::array<float, 2> rect{0.f, 0.f};           // collimation rect (m), 0 = unused
};

struct FieldPrediction {
    std::vector<float>  flux;        // N voxels — joined relative flux
    std::vector<float>  spectrum;    // N * n_bins — per-voxel histogram (row-major)
    std::array<int, 3>  dims{0,0,0}; // D,H,W (volume mode); {N,1,1} (voxel mode)
    int                 n_bins = 0;
    double              inference_ms = 0.0;  // wall-clock of the ONNX Run() call(s)
};

// Hardware-acceleration request. The defaults ask for the best available NVIDIA path:
// the TensorRT EP first (ONNX subgraphs compiled to fused TRT engines, fp16), the CUDA EP
// as a fallback for any op TRT does not claim, and CPU if neither is present — all behind
// one Session, chosen per-subgraph by ONNX Runtime. The first build of a TRT engine is slow
// (seconds–minutes); `engine_cache_dir` persists them so only the first process pays it.
struct ExecutionOptions {
    bool        use_gpu      = true;  // false -> CPU only
    bool        use_tensorrt = true;  // prefer the TensorRT EP (ignored if !use_gpu)
    bool        fp16         = true;  // allow TensorRT fp16 kernels
    int         device_id    = 0;     // CUDA/TensorRT device ordinal
    std::string engine_cache_dir;     // empty -> a default dir under the system temp path
    // Caller-provided CUDA compute stream (a cudaStream_t as an opaque void*). When set, the CUDA /
    // TensorRT EP enqueues its work on THIS stream instead of an internally owned one, so a caller
    // whose input buffers are produced on the same stream needs no cross-stream synchronisation. The
    // Python binding accepts torch.cuda.current_stream().cuda_stream. nullptr -> ORT owns the stream.
    void*       user_compute_stream = nullptr;
};

// The encoded beam that conditions per-voxel prediction. Produced ONCE by encode_beam() and
// reused across many predict_voxelwise() / predict_visible_voxels() calls — the
// (potentially expensive) beam→latent pass runs once and is cached. With a dedicated beam-encoder
// graph attached it holds that encoder's latent output; for a single-graph model it carries the
// beam parameters through so the trunk binds them directly.
struct EncodedBeam {
    BeamParameters     beam;              // source beam (bound directly when no encoder graph)
    std::vector<float> latent;            // beam-encoder output (when an encoder is attached)
    bool               is_encoded = false;
};

// Which predictor a model loads as (each instance reports its own).
enum class PredictorType { VolumeField, VoxelField };

// Device-resident inference outputs: the ONNX flux/spectrum tensors left in CUDA DEVICE memory (no host
// download), produced by VolumeFieldPredictor::predict_to_device. Opaque (defined in the .cpp; it owns the
// bound Ort::Values that back the device buffers — keep it alive until a consumer has copied the data, e.g.
// rfnn::cuda_vk has cudaMemcpy3D'd flux into a Vulkan image). The accessors return raw CUDA device pointers
// (do NOT dereference on the host). This is the zero-copy path; the CPU/fallback path is predict_into_field.
struct DeviceFieldOutputs;
void         release_device_outputs(DeviceFieldOutputs* outputs);
const void*  device_outputs_flux(const DeviceFieldOutputs* outputs);       // device ptr: N scalars
const void*  device_outputs_spectrum(const DeviceFieldOutputs* outputs);   // device ptr: N*bins (or null)
size_t       device_outputs_voxel_count(const DeviceFieldOutputs* outputs);
bool         device_outputs_is_fp16(const DeviceFieldOutputs* outputs);    // flux element type

// The model's INPUT tube-spectrum histogram layout, in eV — everything needed to reconstruct the
// energy binning the model expects: `bins` values spanning [min_energy_ev, max_energy_ev], each
// `bin_width_ev` wide (bin i covers [min + i*width, min + (i+1)*width)). Sourced from the RF3M
// ModelDomain's "spectrum" beam parameter (a Spectrum range). `bins == 0` means no spectrum input.
struct SpectrumInputLayout {
    int   bins          = 0;
    float min_energy_ev = 0.f;
    float max_energy_ev = 0.f;
    float bin_width_ev  = 0.f;
};

// The unit a caller's buffer is expressed in. A bound buffer is fed to the graph VERBATIM, so it
// must hold NORMALISED values; ParameterNormalizer converts a metric buffer into that space using
// the model's own domain, and writes the result into the bound buffer.
enum class Unit : std::uint8_t {
    Normalized = 0,   // already in the network's space — copied through
    Metres,
    Millimetres,
    Degrees,
    Radians,
};

class VolumeFieldPredictor;

// Produced BY a model (`pred.parameter_normalizer()`), because only the model knows the ranges its
// inputs were normalised against — the metric [min,max] per beam parameter and the metric field box
// the [0,1]³ positions map into. Both come from the package's ModelDomain.
//
//     auto norm = pred.parameter_normalizer();
//     norm.write(ModelInput::SourceOrigin3D, origin_m, 3, Unit::Metres);   // -> the bound buffer
//
// The per-query POSITIONS are not here: they are generated internally from set_voxel_grid().
class ParameterNormalizer {
public:
    // Convert `count` values from `unit` into the network's normalised space and RETURN them. The
    // caller decides where they go — upload them to a device buffer with its own framework, or hand
    // them to write() below. Nothing is bound, nothing is touched.
    std::vector<float> normalize(ModelInput flag, const float* values, std::size_t count,
                                 Unit unit = Unit::Metres) const;

    // Same conversion, written straight into the buffer bound for `flag`. HOST buffers only: the
    // runtime does not write device memory from the CPU. For a device-resident input, use
    // normalize() and upload the result yourself.
    void write(ModelInput flag, const float* values, std::size_t count,
               Unit unit = Unit::Metres) const;

private:
    friend class VolumeFieldPredictor;
    explicit ParameterNormalizer(const VolumeFieldPredictor* p) : pred_(p) {}
    static const std::string& graph_name_for(ModelInput);
    const VolumeFieldPredictor* pred_;
};

// VolumeFieldPredictor — runs ONE exported ONNX graph (the model trunk) through ONNX Runtime.
// Field-wise models emit the whole D×H×W volume in a single Run(). Base of the hierarchy.
class VolumeFieldPredictor {
public:
    // Loads `onnx_path`. With use_cuda the GPU execution providers (TensorRT→CUDA) are requested
    // and ONNX Runtime falls back to CPU if unavailable. Inspecting the graph's inputs sets
    // is_voxelwise() (a per-point "position" input => per-voxel model).
    explicit VolumeFieldPredictor(const std::string& onnx_path, bool use_cuda = true);

    // Load an ONNX model that lives in memory (e.g. unpacked from a model package) — the graph
    // never touches disk. `onnx_bytes`/`n` are the serialized ONNX protobuf.
    VolumeFieldPredictor(const void* onnx_bytes, size_t n, bool use_cuda = true);

    // Full control over the execution providers (TensorRT/CUDA/CPU, fp16, engine cache).
    VolumeFieldPredictor(const std::string& onnx_path, const ExecutionOptions& exec);
    VolumeFieldPredictor(const void* onnx_bytes, size_t n, const ExecutionOptions& exec);

    virtual ~VolumeFieldPredictor();

    virtual PredictorType type() const { return PredictorType::VolumeField; }
    bool is_voxelwise() const { return voxelwise_; }   // trunk graph has a per-point position input
    int  spectrum_bins() const { return out_bins_; }   // OUTPUT per-voxel histogram bins (graph output)
    // INPUT beam-spectrum length the graph's "spectrum" input requires (0 if the graph has no such
    // input). Read straight from the ONNX graph by introspect(), so it is the ground truth regardless
    // of what the RF3M ModelDomain metadata claims.
    // Virtual: a two-graph model consumes the spectrum in its BEAM ENCODER, not its trunk, so the
    // per-voxel override forwards there. A caller sizes its beam spectrum from this without knowing
    // whether the package is one graph or two.
    virtual int input_spectrum_bins() const { return in_spectrum_bins_; }
    // The training dataset's metric field box (metres) the normalised [0,1]^3 positions map into.
    // {0,0,0} if the package predates field-dimension metadata. Carried by the RF3M ModelDomain.
    const std::array<float, 3>& field_dimensions() const { return domain_.field_dimensions_m; }

    // The input tube-spectrum binning (bins + min/max energy + bin width, in eV) the model expects,
    // reconstructed from the ModelDomain "spectrum" beam parameter. Use this to build a beam spectrum
    // of the right length and energy mapping regardless of the model's graph topology.
    SpectrumInputLayout input_spectrum_layout() const;
    // True when the ONNX graph emits fp16 outputs — predict_into_field then builds the field's flux
    // layer as fp16 (RadFiled3D float16) instead of float32.
    bool predicts_fp16() const { return out_fp16_; }

    // The execution provider this session actually runs on, decided at load time from which EP successfully
    // registered (GPU EPs are best-effort and fall back): "TensorRT" or "CUDA" => GPU, "CPU" => CPU-only.
    // Use this to report whether inference is hardware-accelerated (a use_gpu request can still land on CPU
    // when the CUDA/TensorRT runtime libraries are missing).
    std::string execution_provider() const;
    bool uses_gpu() const;   // true unless the session fell back to the CPU provider

    // ── zero-copy binding: the caller owns the memory, the model writes into it ──────────────────
    // The buffers are bound ONCE and re-used across inferences; predict_volume() then takes no
    // arguments, because everything it needs has already been announced:
    //
    //     pred.set_voxel_grid({32, 32, 32});
    //     pred.bind_global_parameter(ModelInput::TubeSpectrum, spectrum);   // per-field inputs
    //     pred.bind_output_layer(ModelOutput::Flux, flux);                  // caller's memory
    //     pred.predict_volume();                                            // -> lands in `flux`
    //
    // Bound inputs are read afresh on every predict_volume(), so writing new values into a bound
    // buffer in place is picked up by the next inference — no rebinding. Values are fed to the graph
    // VERBATIM: the buffer must already hold what the network was trained on (normalised), unlike
    // the BeamParameters path, which normalises metric values against the domain ranges for you.
    //
    // convert_buffer = false (the default) is STRICT: the buffer's device and precision must be what
    // the session consumes, or the bind throws BindingError explaining the mismatch and its cost.
    // convert_buffer = true accepts the mismatch and stages the buffer — which for an input SNAPSHOTS
    // it at bind time (later in-place edits are NOT seen) and for an output copies back after each run.
    void bind_global_parameter(ModelInput flag, const MemoryRef& buffer, bool convert_buffer = false);
    void bind_output_layer(ModelOutput flag, const MemoryRef& buffer, bool convert_buffer = false);
    void clear_bindings();

    // The grid the next predict_volume() fills. Chosen per call by the caller (the C++ runtime sizes
    // it to available GPU memory); nothing about it is baked into the graph.
    void set_voxel_grid(std::array<int, 3> voxel_counts);
    std::array<int, 3> voxel_grid() const;

    // Run into the bound buffers. Throws IncompleteSetupError, listing EVERY gap at once, if the grid
    // or any required binding is missing.
    virtual void predict_volume();

    // What this session can consume with no copy (device / precision / provider) — the yardstick the
    // bind_* calls check against.
    SessionMemory session_memory() const;

    // The number of elements a buffer for `flag` must hold at the current grid. Bind by CAPACITY:
    // size for the largest grid you will request, then vary the grid per frame.
    std::size_t required_elements(ModelOutput flag) const;
    std::size_t required_elements(ModelInput flag) const;

    // A converter that knows THIS model's domain: metric values in, normalised values written into
    // the bound buffers. See ParameterNormalizer.
    ParameterNormalizer parameter_normalizer() const;

    // Register the [min,max] metric range of a beam parameter (taken from the RF3M ModelDomain).
    // Parameters the model trained on in NORMALIZED form — i.e. a metric input it does not normalise
    // itself (e.g. "distance" in metres, "opening_angle" in degrees) — are clipped to this range and
    // linearly mapped to [0,1] before encoding, matching the training-time BeamParametersNormalization.
    // Self-normalised inputs (the "direction" unit vector, the "spectrum" histogram) are left untouched.
    // rfnn::io::V1::ModelStore::load() calls this for every domain parameter; without it, metric
    // inputs reach the graph un-normalised and the prediction is wrong (the deployed beam latent does
    // not match training).
    void set_parameter_range(const std::string& name, float min, float max);

    // ── RF3M package metadata ───────────────────────────────────────────────────────────────────
    // Populated by rfnn::io::V1::ModelStore::load[/_from_memory] from the package the predictor
    // was loaded from (so the factory returns the runnable predictor directly, carrying its own
    // domain/provenance/metrics + the names of the graphs it was composed from). A predictor built
    // straight from a bare ONNX path/buffer leaves these empty. The factory is the only writer.
    // What this model consumes/produces, as DECLARED in the package it was loaded from. A consumer
    // asks the interface (does it take a translation? does it emit a spectrum?) instead of guessing
    // from the graph's tensor names.
    const radfield3dnn::ModelInterface& interface() const { return interface_; }
    const rfnn::io::ModelDomain&        domain() const { return domain_; }
    const rfnn::io::ModelProvenance&    provenance() const { return provenance_; }
    const std::map<std::string, float>& metrics() const { return metrics_; }
    const std::vector<std::string>&     graph_names() const { return graph_names_; }
    void set_package_metadata(rfnn::io::ModelDomain domain, rfnn::io::ModelProvenance provenance,
                              std::map<std::string, float> metrics,
                              std::vector<std::string> graph_names,
                              radfield3dnn::ModelInterface interface);

    // Whole-field prediction. Field-wise models emit the volume in one Run(); VoxelFieldPredictor
    // overrides this to tile per-voxel queries over the grid in `max_inner_batch` chunks.
    virtual FieldPrediction predict_volume(const BeamParameters& beam, std::array<int, 3> dims,
                                           int max_inner_batch = 65536) const;

    // Predict straight into `out`'s prediction channel (creating the "flux"/"spectrum" layers if
    // absent), writing the ONNX outputs directly into the field's buffers with no intermediate
    // copy. `out`'s voxel counts set the resolution. VoxelFieldPredictor overrides this to tile.
    // This is the HOST / CPU-fallback path (outputs land in the field's host layer buffers).
    virtual void predict_into_field(const BeamParameters& beam, DeviceCartesianRadiationField& out,
                                    int max_inner_batch = 65536) const;

    // Device-resident whole-field inference: run the graph with the ONNX outputs bound to CUDA DEVICE
    // memory (IoBinding) — nothing is downloaded to the host. Returns an opaque DeviceFieldOutputs holding
    // the flux/spectrum device pointers (and keeping the device buffers alive), or NULLPTR when there is no
    // GPU support. A field-wise (volume) model runs in one Run; VoxelFieldPredictor overrides this to tile
    // per-voxel queries into ONE device buffer (same N-value output). Caller release_device_outputs() it.
    virtual DeviceFieldOutputs* predict_to_device(const BeamParameters& beam, std::array<int, 3> dims,
                                                  int device_id = 0) const;

    // Run this graph on beam parameters (one row) and return its first output flat. Used when this
    // predictor is the beam-encoder of a VoxelFieldPredictor (beam parameters -> latent vector).
    std::vector<float> run_field_raw(const BeamParameters& beam) const;

    // Opaque ORT state. Public only so the .cpp's file-scope run helpers can name the type.
    struct Impl;

protected:
    VolumeFieldPredictor(VolumeFieldPredictor&&) noexcept;  // move-adopt (factory composes types)
    void introspect();                 // populate voxelwise_/out_bins_ from the loaded graph

    // Every gap in the setup, reported in ONE message (grid, unbound inputs, no output).
    void check_setup_complete() const;

    friend class ParameterNormalizer;
    friend class VoxelFieldPredictor;   // drives its beam-encoder sub-predictor's session

    // Post-run output sinks — invoked once at the tail of every predict_volume()/predict_voxels(),
    // after the outputs are written and synchronised. rfnn::vk::VulkanImageTarget registers one to
    // relayout its CUDA staging buffer into the bound Vulkan image on each run. Cleared by
    // clear_bindings(). Registration is private (friend) to keep the surface minimal.
    friend class rfnn::vk::VulkanImageTarget;
    void add_output_sink(std::function<void()> sink);

    std::unique_ptr<Impl> impl_;
    bool voxelwise_ = false;
    int  out_bins_  = 32;
    int  in_spectrum_bins_ = 0;   // length of the graph's "spectrum" input (set by introspect())
    bool out_fp16_  = false;   // ONNX graph emits fp16 outputs (set by introspect())

    // RF3M package metadata (empty unless set by the factory). Plain members so the defaulted
    // move-adopt ctor carries them when the factory wraps a built trunk into a VoxelFieldPredictor.
    rfnn::io::ModelDomain        domain_;
    rfnn::io::ModelProvenance    provenance_;
    std::map<std::string, float> metrics_;
    std::vector<std::string>     graph_names_;
    radfield3dnn::ModelInterface interface_;
};

// VoxelFieldPredictor — a per-voxel implicit model. IS-A VolumeFieldPredictor (it assembles a whole
// volume by tiling), and additionally answers single-voxel queries and projection-limited
// sub-volumes. The trunk takes (position, conditioning); the conditioning is a latent from a
// separate beam-encoder graph (or the beam parameters bound directly when no encoder is attached).
class VoxelFieldPredictor : public VolumeFieldPredictor {
public:
    // The factory passes the prediction trunk (loaded into the base) and the beam-encoder graph
    // (may be null -> the trunk binds the beam parameters directly, single-graph behaviour).
    VoxelFieldPredictor(const void* trunk_bytes, size_t n,
                        std::shared_ptr<VolumeFieldPredictor> beam_encoder,
                        bool use_cuda = true);
    VoxelFieldPredictor(const void* trunk_bytes, size_t n,
                        std::shared_ptr<VolumeFieldPredictor> beam_encoder,
                        const ExecutionOptions& exec);
    // Adopt an already-built trunk (the factory builds it once, then wraps it without re-loading).
    VoxelFieldPredictor(VolumeFieldPredictor&& trunk,
                        std::shared_ptr<VolumeFieldPredictor> beam_encoder);

    PredictorType type() const override { return PredictorType::VoxelField; }

    // Encode the beam ONCE into the latent that conditions per-voxel prediction; cache + reuse.
    EncodedBeam encode_beam(const BeamParameters& beam) const;

    // Per-voxel models reach the volume by tiling: the bound output buffers are filled chunk by
    // chunk, each Run() writing straight into its own offset of the caller's memory.
    void predict_volume() override;

    // Predict ONLY these voxels of the bound grid, writing each one to ITS OWN location in the bound
    // output buffers. Every other voxel is left exactly as the caller had it — so a renderer can
    // refresh a moving ROI, or progressively fill a field, without recomputing (or clearing) the
    // rest. Indices are (i, j, k) into the grid set by set_voxel_grid(); flat offset = i + D*j + D*H*k
    // (x-FASTEST — matches RadFiled3D's VoxelGrid layout and the renderer's texture layout).
    //
    // Still zero-copy: the voxels are sorted and grouped into contiguous runs, and each run's Run()
    // binds the caller's memory AT that run's offset. A compact ROI (a slab, a box) is therefore one
    // Run per row; a fully scattered set costs one Run per isolated voxel.
    void predict_voxels(const std::vector<std::array<int, 3>>& voxel_indices);

private:
    // Run the beam-encoder graph over the BOUND global buffers. Re-run per inference, which is what
    // makes an in-place edit of a bound input visible without rebinding.
    std::vector<float> encode_bound_beam() const;
    // Capture the trunk's real output shapes once (the exported graph declares none).
    void ensure_output_layout(const std::vector<float>& latent) const;

    // ── GPU-session bound path (device-resident latent + positions) ──────────────────────────────
    // On a CUDA session these keep the beam latent and the position grid in DEVICE memory across the
    // tiled runs, so nothing crosses the PCIe bus per chunk. Each returns false when the device
    // data-transfer is unavailable (CPU session, or the CUDA provider library will not register), and
    // the caller falls back to the host broadcast path — same result, more traffic.
    bool predict_volume_device();
    bool predict_voxels_device(const std::vector<std::size_t>& flat_offsets);
    // Run the beam-encoder over the bound globals, leave the latent on the device, and broadcast it to
    // `chunk_rows` rows of the persistent device buffer. Sets the cached latent width.
    void encode_broadcast_device(int chunk_rows);
    // One tiled run of `rows` voxels starting at flat `offset`: position + latent inputs are device
    // sub-views, outputs are the caller's (device) buffers bound at the run's offset.
    void run_span_device(std::size_t offset, int rows, int latent_width);
public:

    // The beam encoder is the graph that consumes the tube spectrum in a two-graph package.
    int input_spectrum_bins() const override {
        return beam_encoder_ ? beam_encoder_->input_spectrum_bins()
                             : VolumeFieldPredictor::input_spectrum_bins();
    }

    // Per-voxel query. `positions`: M points in normalised [0,1]^3; `beam`: a cached EncodedBeam.
    FieldPrediction predict_voxelwise(const std::vector<std::array<float, 3>>& positions,
                                      const EncodedBeam& beam) const;

    // Per-voxel query with ABSOLUTE positions in metres: each xyz is normalised by the model's field
    // dimensions (domain().field_dimensions_m, the training dataset's box in metres) into [0,1]^3, then
    // forwarded to predict_voxelwise. An unknown (0) field dimension leaves that axis untouched.
    FieldPrediction predict_voxelwise_absolute(const std::vector<std::array<float, 3>>& positions_m,
                                               const EncodedBeam& beam) const;
    // Same, over a CONTIGUOUS [count, 3] (x,y,z) buffer bound DIRECTLY as the ONNX position input
    // (no copy) — e.g. vulkan::VoxelVisibilityCuller::cull(...).positions, for a zero-copy GPU-cull
    // → ONNX hand-off. The buffer must stay alive for the call.
    FieldPrediction predict_voxelwise(const float* positions_xyz, size_t count,
                                      const EncodedBeam& beam) const;

    // Only the voxels whose centre projects inside `projection` (column-major 4×4 mapping the
    // normalised [0,1]^3 position to clip; kept when x,y ∈ [-1,1], z ∈ [0,1], w > 0) — the
    // sub-volume that projection sees. CPU visibility test; for the GPU compute-shader version see
    // compute_visible_voxels() in vk/vulkan_field.h, then feed its result to predict_voxelwise().
    FieldPrediction predict_visible_voxels(const EncodedBeam& beam, std::array<int, 3> dims,
                                                   const std::array<float, 16>& projection) const;

    // Whole-field: encode the beam once, then tile predict_voxelwise over the grid.
    FieldPrediction predict_volume(const BeamParameters& beam, std::array<int, 3> dims,
                                   int max_inner_batch = 65536) const override;

    // Whole-field: tile per-voxel queries straight into `out`'s buffers (zero intermediate copy).
    void predict_into_field(const BeamParameters& beam, DeviceCartesianRadiationField& out,
                            int max_inner_batch = 65536) const override;

    // Device-resident whole-field: tile per-voxel queries with each chunk's ONNX output bound to CUDA
    // device memory, copied into ONE device flux buffer (the same N-value output a volume model yields).
    // Returns null without the CUDA interop. Caller release_device_outputs() it.
    DeviceFieldOutputs* predict_to_device(const BeamParameters& beam, std::array<int, 3> dims,
                                          int device_id = 0) const override;
    // Predict ONLY `voxel_locations` (integer (i,j,k) voxel indices): the written field holds the
    // predicted values at those voxels and -inf (flux and every spectrum bin) at all others.
    void predict_into_field(const BeamParameters& beam, DeviceCartesianRadiationField& out,
                            const std::vector<std::array<int, 3>>& voxel_locations,
                            int max_inner_batch = 65536) const;

private:
    // Tile the grid and write predictions straight into `flux_dst` (N floats) and `spec_dst`
    // (N*bins floats) — either may be null to skip. The single per-voxel assembly path, shared by
    // predict_volume() (into a FieldPrediction) and predict_into_field() (into the field buffers).
    void tile_into(const EncodedBeam& beam, std::array<int, 3> dims, int max_inner_batch,
                   float* flux_dst, float* spec_dst) const;

    std::shared_ptr<VolumeFieldPredictor> beam_encoder_;
};

}  // namespace radfield3dnn
