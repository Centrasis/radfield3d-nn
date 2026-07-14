// Python bindings for the DEPLOYMENT runtime (rfnn::io::V1::ModelStore + the ONNX field
// predictors). No CUDA / torch / tcnn dependency — this is the pure ONNX-Runtime half, so a
// trained RF3M package can be loaded and executed from Python exactly as the C++ deployment
// would run it (the Python-side test of the deploy path).
//
//   import rfnn_deploy
//   pred = rfnn_deploy.ModelStore.load("PBRFNet.rf3m") # RF3M -> runnable predictor (Voxel|Volume)
//   pred.domain, pred.metrics, pred.graph_names          # package metadata, carried on the predictor
//   out  = pred.predict_volume(beam, (48,48,48))         # -> dict(flux=np[D,H,W], spectrum=np[D,H,W,B])
//   enc  = pred.encode_beam(beam)                        # voxel models: beam latent (cached)
//   out  = pred.predict_voxelwise(positions_np, enc)     # per-voxel queries, positions in [0,1]^3
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <radfield3d-nn/field_predictors.h>
#include <radfield3d-nn/model_binding.h>
#include <radfield3d-nn/model_interface.h>
#include <radfield3d-nn/model_io.h>

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using radfield3dnn::BeamParameters;
using radfield3dnn::ModelInput;
using radfield3dnn::ModelInterface;
using radfield3dnn::ModelOutput;
using radfield3dnn::EncodedBeam;
using radfield3dnn::ExecutionOptions;
using radfield3dnn::FieldPrediction;
using radfield3dnn::PredictorType;
using radfield3dnn::VolumeFieldPredictor;
using radfield3dnn::VoxelFieldPredictor;

namespace {

// ── caller memory -> MemoryRef, with NO copy ────────────────────────────────────────────────────
// Accepts a torch.Tensor (CPU or CUDA) or a numpy array. What we need is the raw pointer, its
// capacity, the device it lives on, and its element type — a torch CUDA tensor hands us a device
// pointer, which is exactly what ONNX Runtime binds when the session runs on that device.
radfield3dnn::MemoryRef memory_ref_of(const py::object& o) {
    using radfield3dnn::DeviceKind;
    using radfield3dnn::ElementType;
    radfield3dnn::MemoryRef m;

    auto dtype_from = [](const std::string& name) {
        if (name.find("float32") != std::string::npos || name == "float32") return ElementType::F32;
        if (name.find("float16") != std::string::npos) return ElementType::F16;
        throw radfield3dnn::BindingError(
            "unsupported buffer dtype '" + name + "': the deploy runtime binds float32 or float16.");
    };

    // torch.Tensor — duck-typed, so importing torch is never required to use this module.
    if (py::hasattr(o, "data_ptr") && py::hasattr(o, "device") && py::hasattr(o, "is_contiguous")) {
        if (!o.attr("is_contiguous")().cast<bool>())
            throw radfield3dnn::BindingError(
                "the bound tensor is not contiguous — a strided view has no single buffer to bind. "
                "Pass tensor.contiguous().");
        m.data  = reinterpret_cast<void*>(o.attr("data_ptr")().cast<std::uintptr_t>());
        m.dtype = dtype_from(py::str(o.attr("dtype")).cast<std::string>());
        m.bytes = o.attr("numel")().cast<std::size_t>()
                * o.attr("element_size")().cast<std::size_t>();

        const auto dev = o.attr("device");
        const std::string kind = py::str(dev.attr("type")).cast<std::string>();
        if (kind == "cpu")       m.device = DeviceKind::Cpu;
        else if (kind == "cuda") m.device = DeviceKind::Cuda;
        else if (kind == "vulkan")
            throw radfield3dnn::BindingError(
                "a Vulkan tensor cannot be bound: ONNX Runtime has no Vulkan execution provider, so "
                "no graph can read or write that memory. Run the session on CUDA and bind CUDA "
                "buffers; the CUDA->Vulkan export path shares the result with a Vulkan image without "
                "a host round-trip.");
        else
            throw radfield3dnn::BindingError("unsupported tensor device '" + kind + "'.");

        if (m.device != DeviceKind::Cpu && !dev.attr("index").is_none())
            m.device_id = dev.attr("index").cast<int>();
        return m;
    }

    // numpy array — always host memory.
    if (py::isinstance<py::array>(o)) {
        auto a = py::cast<py::array>(o);
        if (!(a.flags() & py::array::c_style))
            throw radfield3dnn::BindingError(
                "the bound array is not C-contiguous. Pass np.ascontiguousarray(a).");
        m.data   = const_cast<void*>(a.data());
        m.bytes  = static_cast<std::size_t>(a.nbytes());
        m.device = DeviceKind::Cpu;
        m.dtype  = dtype_from(py::str(a.dtype()).cast<std::string>());
        return m;
    }

    throw radfield3dnn::BindingError(
        "expected a torch.Tensor or a numpy array to bind; got '"
        + py::str(o.get_type().attr("__name__")).cast<std::string>() + "'.");
}

// Everything a user needs to see at a glance: which backend is actually executing (a use_gpu request
// can silently fall back to CPU), the grid it will fill, and what the model consumes and produces.
std::string describe_predictor(const radfield3dnn::VolumeFieldPredictor& p) {
    const auto& iface = p.interface();
    const auto  sm    = p.session_memory();
    const auto  g     = p.voxel_grid();

    auto flags = [&] {
        std::string in, out;
        for (std::uint32_t b = 0; b < 32; ++b) {
            const auto f = static_cast<radfield3dnn::ModelInput>(1u << b);
            if (iface.takes(f)) in += (in.empty() ? "" : "|") + radfield3dnn::to_string(f);
            const auto o = static_cast<radfield3dnn::ModelOutput>(1u << b);
            if (iface.gives(o)) out += (out.empty() ? "" : "|") + radfield3dnn::to_string(o);
        }
        return std::make_pair(in.empty() ? "-" : in, out.empty() ? "-" : out);
    }();

    std::ostringstream o;
    o << "<" << (p.type() == radfield3dnn::PredictorType::VoxelField ? "VoxelFieldPredictor"
                                                                     : "VolumeFieldPredictor")
      << "\n  backend      : " << sm.provider << " (" << radfield3dnn::to_string(sm.device);
    if (sm.device != radfield3dnn::DeviceKind::Cpu) o << ":" << sm.device_id;
    o << ", " << radfield3dnn::to_string(sm.dtype) << ")"
      << "\n  interface    : " << std::hex << "0x" << iface.id() << std::dec
      << "\n    inputs     : " << flags.first
      << "\n    outputs    : " << flags.second
      << "\n  voxel_grid   : ";
    if (g[0] > 0) o << g[0] << "x" << g[1] << "x" << g[2];
    else          o << "(unset — call set_voxel_grid)";
    o << "\n  spectrum     : " << p.input_spectrum_bins() << " in -> " << p.spectrum_bins() << " out bins"
      << "\n  field box (m): " << p.field_dimensions()[0] << ", " << p.field_dimensions()[1]
      << ", " << p.field_dimensions()[2]
      << "\n  graphs       : ";
    for (const auto& n : p.graph_names()) o << n << " ";
    o << "\n>";
    return o.str();
}

py::dict prediction_to_dict(const FieldPrediction& fp) {
    py::dict d;
    const auto [D, H, W] = fp.dims;
    // flux: volume mode (D,H,W); voxel mode (N,)
    if (H == 1 && W == 1) {
        d["flux"] = py::array_t<float>({(py::ssize_t)D}, fp.flux.data());
        if (fp.n_bins > 0 && !fp.spectrum.empty())
            d["spectrum"] = py::array_t<float>({(py::ssize_t)D, (py::ssize_t)fp.n_bins}, fp.spectrum.data());
    } else {
        d["flux"] = py::array_t<float>({(py::ssize_t)D, (py::ssize_t)H, (py::ssize_t)W}, fp.flux.data());
        if (fp.n_bins > 0 && !fp.spectrum.empty())
            d["spectrum"] = py::array_t<float>({(py::ssize_t)D, (py::ssize_t)H, (py::ssize_t)W,
                                                (py::ssize_t)fp.n_bins}, fp.spectrum.data());
    }
    d["dims"] = py::make_tuple(D, H, W);
    d["n_bins"] = fp.n_bins;
    d["inference_ms"] = fp.inference_ms;
    return d;
}

BeamParameters make_beam(std::array<float, 3> direction, std::array<float, 3> origin,
                         std::vector<float> spectrum, std::array<float, 2> rect) {
    BeamParameters b;
    b.direction = direction;
    b.origin = origin;
    b.spectrum = std::move(spectrum);
    b.rect = rect;
    return b;
}

}  // namespace

PYBIND11_MODULE(rfnn_deploy, m) {
    // Deployment interface: WHAT a stored model consumes / produces. Bit flags (append-only ABI,
    // see model_interface.h); combine with |, test with &. Python does not redeclare these.
    py::enum_<ModelInput>(m, "ModelInput", py::arithmetic())
        .value("NONE", ModelInput::None)
        .value("POSITION", ModelInput::Position)
        .value("QUERY_DIRECTION", ModelInput::QueryDirection)
        .value("TUBE_SPECTRUM", ModelInput::TubeSpectrum)
        .value("BEAM_DIRECTION", ModelInput::BeamDirection)
        .value("SOURCE_DISTANCE", ModelInput::SourceDistance)
        .value("SOURCE_ORIGIN_3D", ModelInput::SourceOrigin3D)
        .value("BEAM_COLLIMATION", ModelInput::BeamCollimation)
        .value("PATIENT_TRANSLATION_3D", ModelInput::PatientTranslation3D)
        .value("PATIENT_ROTATION_3D", ModelInput::PatientRotation3D)
        .value("GEOMETRY_MAP", ModelInput::GeometryMap)
        .value("ANODE_ANGLE", ModelInput::AnodeAngle)
        .def("__or__", [](ModelInput a, ModelInput b) { return a | b; })
        .def("__and__", [](ModelInput a, ModelInput b) {
            return static_cast<ModelInput>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b)); })
        .def("__bool__", [](ModelInput a) { return static_cast<std::uint32_t>(a) != 0u; });

    py::enum_<ModelOutput>(m, "ModelOutput", py::arithmetic())
        .value("NONE", ModelOutput::None)
        .value("FLUX", ModelOutput::Flux)
        .value("SPECTRUM", ModelOutput::Spectrum)
        .value("ANGULAR_FLUX", ModelOutput::AngularFlux)
        .value("ERROR", ModelOutput::Error)
        .value("AIR_KERMA", ModelOutput::AirKerma)
        .value("SEPARATE_CHANNELS", ModelOutput::SeparateChannels)
        .def("__or__", [](ModelOutput a, ModelOutput b) { return a | b; })
        .def("__and__", [](ModelOutput a, ModelOutput b) {
            return static_cast<ModelOutput>(static_cast<std::uint32_t>(a) & static_cast<std::uint32_t>(b)); })
        .def("__bool__", [](ModelOutput a) { return static_cast<std::uint32_t>(a) != 0u; });

    py::class_<ModelInterface>(m, "ModelInterface")
        .def(py::init<>())
        .def_readwrite("inputs", &ModelInterface::inputs)
        .def_readwrite("outputs", &ModelInterface::outputs)
        .def_readwrite("resolution_aware", &ModelInterface::resolution_aware)
        .def_readwrite("region_state_dims", &ModelInterface::region_state_dims)
        .def_readwrite("region_width_frame", &ModelInterface::region_width_frame)
        .def_property_readonly("id", &ModelInterface::id)
        .def_property_readonly("is_voxelwise", &ModelInterface::is_voxelwise)
        .def("validate", &ModelInterface::validate, py::arg("domain"),
             "Reject reserved bits and check the domain carries every resolution the declared flags "
             "need. Called on store and on load.")
        .def("takes", &ModelInterface::takes, py::arg("flag"))
        .def("gives", &ModelInterface::gives, py::arg("flag"))
        .def_static("make_id", &ModelInterface::make_id, py::arg("inputs"), py::arg("outputs"));

    m.doc() = "RadField3D-NN deployment runtime (RF3M + ONNX field predictors) — python bindings";

    py::class_<BeamParameters>(m, "BeamParameters")
        .def(py::init(&make_beam),
             py::arg("direction"), py::arg("origin") = std::array<float, 3>{0.5f, 0.5f, 0.5f},
             py::arg("spectrum") = std::vector<float>{}, py::arg("rect") = std::array<float, 2>{0.f, 0.f})
        .def_readwrite("direction", &BeamParameters::direction)
        .def_readwrite("origin", &BeamParameters::origin)
        .def_readwrite("spectrum", &BeamParameters::spectrum)
        .def_readwrite("rect", &BeamParameters::rect);

    py::class_<ExecutionOptions>(m, "ExecutionOptions")
        .def(py::init<>())
        .def_readwrite("use_gpu", &ExecutionOptions::use_gpu)
        .def_readwrite("use_tensorrt", &ExecutionOptions::use_tensorrt)
        .def_readwrite("fp16", &ExecutionOptions::fp16)
        .def_readwrite("device_id", &ExecutionOptions::device_id)
        .def_readwrite("engine_cache_dir", &ExecutionOptions::engine_cache_dir)
        // The caller's CUDA stream, passed as an integer handle (torch:
        // torch.cuda.current_stream().cuda_stream). The EP then enqueues on it, so buffers the caller
        // produced on that stream need no cross-stream sync. 0 -> ORT owns the stream.
        .def_property("user_compute_stream",
            [](const ExecutionOptions& e) {
                return reinterpret_cast<std::uintptr_t>(e.user_compute_stream); },
            [](ExecutionOptions& e, std::uintptr_t s) {
                e.user_compute_stream = reinterpret_cast<void*>(s); });

    py::class_<EncodedBeam>(m, "EncodedBeam")
        .def_readonly("is_encoded", &EncodedBeam::is_encoded)
        .def_readonly("latent", &EncodedBeam::latent);

    py::enum_<radfield3dnn::Backend>(m, "Backend")
        .value("Cpu", radfield3dnn::Backend::Cpu)
        .value("Cuda", radfield3dnn::Backend::Cuda)
        .value("TensorRT", radfield3dnn::Backend::TensorRT);
    m.def("backend_from_string", &radfield3dnn::backend_from_string, py::arg("name"),
          "Parse 'cpu' / 'cuda' / 'tensorrt'. The vocabulary is defined in C++ so both languages "
          "accept the same spellings and raise the same error.");

    py::enum_<radfield3dnn::DeviceKind>(m, "DeviceKind")
        .value("Cpu", radfield3dnn::DeviceKind::Cpu)
        .value("Cuda", radfield3dnn::DeviceKind::Cuda)
        .value("Vulkan", radfield3dnn::DeviceKind::Vulkan);

    py::enum_<radfield3dnn::ElementType>(m, "ElementType")
        .value("F32", radfield3dnn::ElementType::F32)
        .value("F16", radfield3dnn::ElementType::F16);

    // The unit a caller's buffer is in; ParameterNormalizer maps it into the network's space.
    py::enum_<radfield3dnn::Unit>(m, "Unit")
        .value("Normalized", radfield3dnn::Unit::Normalized)
        .value("Metres", radfield3dnn::Unit::Metres)
        .value("Millimetres", radfield3dnn::Unit::Millimetres)
        .value("Degrees", radfield3dnn::Unit::Degrees)
        .value("Radians", radfield3dnn::Unit::Radians);

    py::class_<radfield3dnn::SessionMemory>(m, "SessionMemory")
        .def_readonly("device", &radfield3dnn::SessionMemory::device)
        .def_readonly("dtype", &radfield3dnn::SessionMemory::dtype)
        .def_readonly("device_id", &radfield3dnn::SessionMemory::device_id)
        .def_readonly("provider", &radfield3dnn::SessionMemory::provider);

    py::class_<radfield3dnn::ParameterNormalizer>(m, "ParameterNormalizer")
        .def("normalize",
             [](const radfield3dnn::ParameterNormalizer& self, ModelInput flag,
                const py::object& values, radfield3dnn::Unit unit) {
                 auto a = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(values);
                 auto out = self.normalize(flag, a.data(), static_cast<std::size_t>(a.size()), unit);
                 return py::array_t<float>(static_cast<py::ssize_t>(out.size()), out.data());
             },
             py::arg("flag"), py::arg("values"), py::arg("unit") = radfield3dnn::Unit::Metres,
             "Convert metric values into the network's normalised space (using this model's own "
             "domain) and RETURN them as a numpy array. Upload them wherever you like — this is the "
             "path to use for a device-resident buffer.")
        .def("write",
             [](const radfield3dnn::ParameterNormalizer& self, ModelInput flag,
                const py::object& values, radfield3dnn::Unit unit) {
                 auto a = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(values);
                 self.write(flag, a.data(), static_cast<std::size_t>(a.size()), unit);
             },
             py::arg("flag"), py::arg("values"), py::arg("unit") = radfield3dnn::Unit::Metres,
             "Convert metric values into the network's normalised space (using this model's own "
             "domain) and write them into the buffer bound for `flag`.");

    py::register_exception<radfield3dnn::BindingError>(m, "BindingError");
    py::register_exception<radfield3dnn::IncompleteSetupError>(m, "IncompleteSetupError");

    py::enum_<PredictorType>(m, "PredictorType")
        .value("VolumeField", PredictorType::VolumeField)
        .value("VoxelField", PredictorType::VoxelField);

    // Reconstructed input tube-spectrum binning (eV).
    py::class_<radfield3dnn::SpectrumInputLayout>(m, "SpectrumInputLayout")
        .def_readonly("bins", &radfield3dnn::SpectrumInputLayout::bins)
        .def_readonly("min_energy_ev", &radfield3dnn::SpectrumInputLayout::min_energy_ev)
        .def_readonly("max_energy_ev", &radfield3dnn::SpectrumInputLayout::max_energy_ev)
        .def_readonly("bin_width_ev", &radfield3dnn::SpectrumInputLayout::bin_width_ev);

    // The package metadata is carried ON the predictor (set by ModelStore::load), exposed as
    // read-only properties .domain / .provenance / .metrics / .graph_names (inherited by
    // VoxelFieldPredictor). dynamic_attr stays so callers may still attach their own attributes.
    py::class_<VolumeFieldPredictor, std::shared_ptr<VolumeFieldPredictor>>(m, "VolumeFieldPredictor", py::dynamic_attr())
        .def(py::init<const std::string&, bool>(), py::arg("onnx_path"), py::arg("use_cuda") = false)
        .def_property_readonly("type", &VolumeFieldPredictor::type)
        .def_property_readonly("is_voxelwise", &VolumeFieldPredictor::is_voxelwise)
        .def_property_readonly("spectrum_bins", &VolumeFieldPredictor::spectrum_bins)
        .def_property_readonly("input_spectrum_bins", &VolumeFieldPredictor::input_spectrum_bins)
        .def_property_readonly("input_spectrum_layout", &VolumeFieldPredictor::input_spectrum_layout)
        .def_property_readonly("field_dimensions_m", &VolumeFieldPredictor::field_dimensions)
        .def_property_readonly("interface", &VolumeFieldPredictor::interface,
                               "The I/O the package DECLARES (rfnn_deploy.ModelInterface).")
        .def_property_readonly("domain", &VolumeFieldPredictor::domain)
        .def_property_readonly("provenance", &VolumeFieldPredictor::provenance)
        .def_property_readonly("metrics", &VolumeFieldPredictor::metrics)
        .def_property_readonly("graph_names", &VolumeFieldPredictor::graph_names)
        .def("predict_volume",
             [](const VolumeFieldPredictor& self, const BeamParameters& beam,
                std::array<int, 3> dims, int max_inner_batch) {
                 py::gil_scoped_release release;
                 FieldPrediction fp = self.predict_volume(beam, dims, max_inner_batch);
                 py::gil_scoped_acquire acquire;
                 return prediction_to_dict(fp);
             },
             py::arg("beam"), py::arg("dims"), py::arg("max_inner_batch") = 65536)

        // ── zero-copy binding: the caller owns the memory ────────────────────────────────────────
        .def("set_voxel_grid", &VolumeFieldPredictor::set_voxel_grid, py::arg("voxel_counts"),
             "The grid the next predict_volume() fills. Chosen per call — nothing about it is baked "
             "into the graph.")
        .def_property_readonly("voxel_grid", &VolumeFieldPredictor::voxel_grid)
        .def("bind_global_parameter",
             [](VolumeFieldPredictor& self, ModelInput flag, const py::object& buf, bool convert_buffer) {
                 self.bind_global_parameter(flag, memory_ref_of(buf), convert_buffer);
             },
             // keep_alive<self, buffer>: the model holds a raw pointer INTO this array. Without
             // this, binding a temporary (`bind(..., torch.ones(150, device="cuda"))`) frees it the
             // moment the call returns, the allocator recycles the memory, and the next inference
             // silently reads whatever landed there. Holding the reference makes that impossible.
             py::keep_alive<1, 3>(),
             py::arg("flag"), py::arg("buffer"), py::arg("convert_buffer") = false,
             "Bind a per-field input (numpy array or torch.Tensor, CPU or CUDA) for `flag`.\n\n"
             "The buffer is read afresh on every predict_volume(), so updating it IN PLACE is picked "
             "up by the next inference — no rebinding. Values are fed to the graph verbatim: the "
             "buffer must already hold NORMALISED values (use parameter_normalizer() to convert "
             "metric ones).\n\n"
             "convert_buffer=False (default) is strict: a device/precision mismatch with the session "
             "raises BindingError explaining the cost. convert_buffer=True accepts a mismatched "
             "buffer and SNAPSHOTS it at bind time — later in-place edits are then NOT seen.")
        .def("bind_output_layer",
             [](VolumeFieldPredictor& self, ModelOutput flag, const py::object& buf, bool convert_buffer) {
                 self.bind_output_layer(flag, memory_ref_of(buf), convert_buffer);
             },
             py::keep_alive<1, 3>(),   // the model writes into this buffer; it must outlive the binding
             py::arg("flag"), py::arg("buffer"), py::arg("convert_buffer") = false,
             "Bind the caller's memory for an output. The model writes STRAIGHT into it — no result "
             "is allocated and nothing is copied. Call set_voxel_grid() first (it fixes the size). "
             "convert_buffer=True stages a mismatched buffer and copies back after each run.")
        .def("clear_bindings", &VolumeFieldPredictor::clear_bindings)
        .def("predict_volume",
             [](VolumeFieldPredictor& self) {
                 py::gil_scoped_release release;
                 self.predict_volume();     // no arguments: it runs into what was bound
             },
             "Run into the bound buffers. Raises IncompleteSetupError listing every gap if the grid "
             "or a required binding is missing.")
        .def("required_elements",
             [](const VolumeFieldPredictor& self, const py::object& flag) {
                 if (py::isinstance<ModelOutput>(flag))
                     return self.required_elements(flag.cast<ModelOutput>());
                 return self.required_elements(flag.cast<ModelInput>());
             },
             py::arg("flag"),
             "How many elements a buffer for this flag must hold at the current grid.")
        .def_property_readonly("session_memory", &VolumeFieldPredictor::session_memory,
                               "What this session consumes with no copy (device / dtype / provider).")
        .def("parameter_normalizer", &VolumeFieldPredictor::parameter_normalizer,
             "A converter that knows THIS model's domain: metric values in, normalised values "
             "written into the bound buffers.")
        .def("__repr__", [](const VolumeFieldPredictor& self) { return describe_predictor(self); });

    // (VoxelFieldPredictor adds sparse prediction into the bound grid.)
    py::class_<VoxelFieldPredictor, VolumeFieldPredictor,
               std::shared_ptr<VoxelFieldPredictor>>(m, "VoxelFieldPredictor", py::dynamic_attr())
        .def("encode_beam", &VoxelFieldPredictor::encode_beam, py::arg("beam"))
        .def("predict_voxels",
             [](VoxelFieldPredictor& self, py::array_t<int, py::array::c_style | py::array::forcecast> idx) {
                 if (idx.ndim() != 2 || idx.shape(1) != 3)
                     throw radfield3dnn::BindingError(
                         "predict_voxels: expected an (N, 3) array of (i, j, k) voxel indices.");
                 std::vector<std::array<int, 3>> v(static_cast<std::size_t>(idx.shape(0)));
                 const int* p = idx.data();
                 for (std::size_t i = 0; i < v.size(); ++i) v[i] = {p[3*i], p[3*i+1], p[3*i+2]};
                 py::gil_scoped_release release;
                 self.predict_voxels(v);
             },
             py::arg("voxel_indices"),
             "Predict ONLY these (i, j, k) voxels of the bound grid, writing each to ITS OWN location "
             "in the bound output buffers. Every other voxel keeps the value the caller had — so a "
             "moving ROI can be refreshed, or a field progressively filled, without recomputing (or "
             "clearing) the rest. Still zero-copy: voxels are grouped into contiguous runs and each "
             "run is bound at its own offset.")
        .def("predict_voxelwise",
             [](const VoxelFieldPredictor& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> positions,
                const EncodedBeam& beam) {
                 if (positions.ndim() != 2 || positions.shape(1) != 3)
                     throw std::invalid_argument("positions must be a (N,3) float array in [0,1]^3");
                 const float* ptr = positions.data();
                 const size_t n = (size_t)positions.shape(0);
                 py::gil_scoped_release release;
                 FieldPrediction fp = self.predict_voxelwise(ptr, n, beam);   // zero-copy bind
                 py::gil_scoped_acquire acquire;
                 return prediction_to_dict(fp);
             },
             py::arg("positions"), py::arg("encoded_beam"))
        .def("predict_voxelwise_absolute",
             [](const VoxelFieldPredictor& self,
                py::array_t<float, py::array::c_style | py::array::forcecast> positions_m,
                const EncodedBeam& beam) {
                 if (positions_m.ndim() != 2 || positions_m.shape(1) != 3)
                     throw std::invalid_argument("positions_m must be a (N,3) float array in metres");
                 std::vector<std::array<float, 3>> pts((size_t)positions_m.shape(0));
                 std::memcpy(pts.data(), positions_m.data(), pts.size() * 3 * sizeof(float));
                 py::gil_scoped_release release;
                 FieldPrediction fp = self.predict_voxelwise_absolute(pts, beam);
                 py::gil_scoped_acquire acquire;
                 return prediction_to_dict(fp);
             },
             py::arg("positions_m"), py::arg("encoded_beam"));

    // ── RF3M container (rfnn::io::V1) ────────────────────────────────────────
    using rfnn::io::V1::ModelStore;

    // Read/WRITE + constructible: these double as the SAVE-side metadata builders, so the Python
    // packager assembles a domain/provenance here and the bytes are produced by the SAME C++
    // serialiser the deploy lib uses (no duplicated byte layout).
    py::enum_<rfnn::io::ParameterRangeType>(m, "ParameterRangeType")
        .value("MinMax", rfnn::io::ParameterRangeType::MinMax)
        .value("Spectrum", rfnn::io::ParameterRangeType::Spectrum)
        .value("Map", rfnn::io::ParameterRangeType::Map);

    // A beam parameter's typed range (tagged variant). Convenience factories build the common kinds;
    // the serialiser stores type + byte length per entry so unknown kinds can be skipped on read.
    py::class_<rfnn::io::ParameterRange> beam_range(m, "ParameterRange");
    beam_range
        .def(py::init<>())
        .def_readwrite("type", &rfnn::io::ParameterRange::type)
        .def_readwrite("min", &rfnn::io::ParameterRange::min)
        .def_readwrite("max", &rfnn::io::ParameterRange::max)
        .def_readwrite("bin_width", &rfnn::io::ParameterRange::bin_width)
        .def_readwrite("unit", &rfnn::io::ParameterRange::unit)
        .def_readwrite("children", &rfnn::io::ParameterRange::children)
        .def_static("min_max", [](float mn, float mx, std::string u) {
                rfnn::io::ParameterRange r; r.type = rfnn::io::ParameterRangeType::MinMax;
                r.min = mn; r.max = mx; r.unit = std::move(u); return r;
            }, py::arg("min"), py::arg("max"), py::arg("unit") = "")
        .def_static("spectrum", [](float mn, float mx, float bw, std::string u) {
                rfnn::io::ParameterRange r; r.type = rfnn::io::ParameterRangeType::Spectrum;
                r.min = mn; r.max = mx; r.bin_width = bw; r.unit = std::move(u); return r;
            }, py::arg("min"), py::arg("max"), py::arg("bin_width"), py::arg("unit") = "")
        .def_static("nested", [](std::vector<std::pair<std::string, rfnn::io::ParameterRange>> ch) {
                rfnn::io::ParameterRange r; r.type = rfnn::io::ParameterRangeType::Map; r.children = std::move(ch); return r;
            }, py::arg("children"));

    py::class_<rfnn::io::BeamParameter>(m, "BeamParameterSpec")
        .def(py::init([](std::string n, rfnn::io::ParameterRange r) {
                 return rfnn::io::BeamParameter{std::move(n), std::move(r)};
             }), py::arg("name"), py::arg("range") = rfnn::io::ParameterRange{})
        .def_readwrite("name", &rfnn::io::BeamParameter::name)
        .def_readwrite("range", &rfnn::io::BeamParameter::range);
    // The KIND of collimation; the parameter width follows from it (see collimation_dims()).
    py::enum_<rfnn::io::CollimationType>(m, "CollimationType")
        .value("None_", rfnn::io::CollimationType::None)
        .value("Rectangle", rfnn::io::CollimationType::Rectangle)
        .value("Cone", rfnn::io::CollimationType::Cone)
        .value("Ellipsoid", rfnn::io::CollimationType::Ellipsoid);

    py::class_<rfnn::io::ModelDomain>(m, "ModelDomain")
        .def(py::init([](int bins, float max_e, std::array<float, 3> field_dims,
                         std::vector<rfnn::io::BeamParameter> bp) {
                 rfnn::io::ModelDomain d; d.spectrum_bins = bins; d.spectrum_max_energy_ev = max_e;
                 d.field_dimensions_m = field_dims; d.beam_parameters = std::move(bp); return d;
             }), py::arg("spectrum_bins") = 0, py::arg("spectrum_max_energy_ev") = 0.f,
             py::arg("field_dimensions_m") = std::array<float, 3>{ {0.f, 0.f, 0.f} },
             py::arg("beam_parameters") = std::vector<rfnn::io::BeamParameter>{})
        .def_readwrite("spectrum_bins", &rfnn::io::ModelDomain::spectrum_bins)
        .def_readwrite("spectrum_max_energy_ev", &rfnn::io::ModelDomain::spectrum_max_energy_ev)
        .def_readwrite("field_dimensions_m", &rfnn::io::ModelDomain::field_dimensions_m)
        .def_readwrite("beam_parameters", &rfnn::io::ModelDomain::beam_parameters)
        // Resolutions the interface flags depend on; validate() checks these are present.
        .def_readwrite("angular_phi_segments", &rfnn::io::ModelDomain::angular_phi_segments)
        .def_readwrite("angular_theta_segments", &rfnn::io::ModelDomain::angular_theta_segments)
        .def_readwrite("collimation", &rfnn::io::ModelDomain::collimation)
        .def_property_readonly("collimation_dims", [](const rfnn::io::ModelDomain& d) {
            return rfnn::io::collimation_dims(d.collimation);  // width of the BeamCollimation tensor
        });
    py::class_<rfnn::io::ModelProvenance>(m, "ModelProvenance")
        .def(py::init([](std::string ds, std::string sw, std::string ph) {
                 return rfnn::io::ModelProvenance{std::move(ds), std::move(sw), std::move(ph)};
             }), py::arg("dataset_name") = "", py::arg("software_version") = "", py::arg("physics") = "")
        .def_readwrite("dataset_name", &rfnn::io::ModelProvenance::dataset_name)
        .def_readwrite("software_version", &rfnn::io::ModelProvenance::software_version)
        .def_readwrite("physics", &rfnn::io::ModelProvenance::physics);

    // ── ModelStore: parses an RF3M package AND builds the runnable predictor in one call (no
    //    LoadedModel handle). The C++ API hands back a unique_ptr<VolumeFieldPredictor> whose
    //    dynamic type may be VoxelFieldPredictor; we hand pybind a shared_ptr built in this TU so
    //    it adopts a single, well-formed control block and downcasts a per-voxel model to
    //    VoxelFieldPredictor via RTTI. The package metadata rides along as the predictor's
    //    `.domain` / `.provenance` / `.metrics` / `.graph_names`. ──
    py::class_<ModelStore>(m, "ModelStore")
        .def_static("load",
                    [](const std::string& path, bool use_cuda)
                        -> std::shared_ptr<VolumeFieldPredictor> {
                        return ModelStore::load(path, use_cuda);
                    },
                    py::arg("path"), py::arg("use_cuda") = false,
                    "Load an RF3M package and return the runnable predictor "
                    "(VoxelFieldPredictor for per-voxel models, VolumeFieldPredictor for field-wise).")
        .def_static("load",
                    [](const std::string& path, const ExecutionOptions& exec)
                        -> std::shared_ptr<VolumeFieldPredictor> {
                        return ModelStore::load(path, exec);
                    },
                    py::arg("path"), py::arg("exec"),
                    "Load with an explicit execution-provider request.")
        .def_static("load",
                    [](const std::string& path, const std::string& backend)
                        -> std::shared_ptr<VolumeFieldPredictor> {
                        return ModelStore::load(path, radfield3dnn::backend_from_string(backend));
                    },
                    py::arg("path"), py::arg("backend"),
                    "Load onto a named backend ('cpu' / 'cuda' / 'tensorrt'). Same function the C++ "
                    "side calls — the backend vocabulary is not redefined in Python.")
        .def_static("load_from_memory",
                    [](py::bytes data, bool use_cuda)
                        -> std::shared_ptr<VolumeFieldPredictor> {
                        std::string s = data;
                        return ModelStore::load_from_memory(s.data(), s.size(), use_cuda);
                    },
                    py::arg("data"), py::arg("use_cuda") = false)
        .def_static("read_graphs",
                    [](const std::string& path) {
                        py::dict out;
                        for (const auto& [name, g] : rfnn::io::V1::ModelStore::read_graphs(path))
                            out[py::str(name)] = py::bytes(reinterpret_cast<const char*>(g.data()), g.size());
                        return out;
                    },
                    py::arg("path"),
                    "Read the raw named ONNX graphs (name -> protobuf bytes) from an RF3M package "
                    "without building a predictor — the read counterpart to save_to_memory.");

    // ── SAVE side (the single source of the RF3M byte layout — used by the Python ModelPackager so
    //    the format is never re-implemented) ──────────────────────────────────────────────────────
    auto to_named_graphs = [](const py::dict& graphs) {
        rfnn::io::V1::NamedGraphs g;
        for (auto kv : graphs) {
            std::string name = py::cast<std::string>(kv.first);
            std::string bytes = py::cast<py::bytes>(kv.second);  // ONNX protobuf bytes
            g[name] = std::vector<uint8_t>(bytes.begin(), bytes.end());
        }
        return g;
    };
    py::class_<ModelStore::PackageMetadata>(m, "PackageMetadata")
        .def_readonly("provenance", &ModelStore::PackageMetadata::provenance)
        .def_readonly("domain", &ModelStore::PackageMetadata::domain)
        .def_readonly("metrics", &ModelStore::PackageMetadata::metrics)
        .def_readonly("interface", &ModelStore::PackageMetadata::interface);

    m.def("read_metadata", &ModelStore::read_metadata, py::arg("path"),
          "Read ONLY the RF3M header (interface + provenance + domain + metrics) — no ONNX Runtime "
          "session, no graph payloads. Use this instead of parsing the container in Python.");

    m.def("save_to_memory",
          [to_named_graphs](const py::dict& graphs, const ModelInterface& interface,
                            const rfnn::io::ModelDomain& domain,
                            const rfnn::io::ModelProvenance& prov,
                            const std::map<std::string, float>& metrics) {
              auto bytes = ModelStore::save_to_memory(to_named_graphs(graphs), interface, domain,
                                                      prov, metrics);
              return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          },
          py::arg("graphs"), py::arg("interface"), py::arg("domain"), py::arg("provenance"),
          py::arg("metrics"),
          "Serialise an RF3M package to bytes (graphs + declared interface + domain + provenance + "
          "metrics). The interface is validated against the domain; a package that promises an "
          "output the domain cannot shape is refused here.");
    m.def("save",
          [to_named_graphs](const std::string& path, const py::dict& graphs,
                            const ModelInterface& interface,
                            const rfnn::io::ModelDomain& domain, const rfnn::io::ModelProvenance& prov,
                            const std::map<std::string, float>& metrics) {
              ModelStore::save(path, to_named_graphs(graphs), interface, domain, prov, metrics);
          },
          py::arg("path"), py::arg("graphs"), py::arg("interface"), py::arg("domain"),
          py::arg("provenance"), py::arg("metrics"),
          "Write an RF3M package straight to disk.");
}
