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
#include <radfield3d-nn/model_io.h>

#include <array>
#include <memory>
#include <stdexcept>

namespace py = pybind11;
using radfield3dnn::BeamParameters;
using radfield3dnn::EncodedBeam;
using radfield3dnn::ExecutionOptions;
using radfield3dnn::FieldPrediction;
using radfield3dnn::PredictorType;
using radfield3dnn::VolumeFieldPredictor;
using radfield3dnn::VoxelFieldPredictor;

namespace {

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
        .def_readwrite("engine_cache_dir", &ExecutionOptions::engine_cache_dir);

    py::class_<EncodedBeam>(m, "EncodedBeam")
        .def_readonly("is_encoded", &EncodedBeam::is_encoded)
        .def_readonly("latent", &EncodedBeam::latent);

    py::enum_<PredictorType>(m, "PredictorType")
        .value("VolumeField", PredictorType::VolumeField)
        .value("VoxelField", PredictorType::VoxelField);

    // The package metadata is carried ON the predictor (set by ModelStore::load), exposed as
    // read-only properties .domain / .provenance / .metrics / .graph_names (inherited by
    // VoxelFieldPredictor). dynamic_attr stays so callers may still attach their own attributes.
    py::class_<VolumeFieldPredictor, std::shared_ptr<VolumeFieldPredictor>>(m, "VolumeFieldPredictor", py::dynamic_attr())
        .def(py::init<const std::string&, bool>(), py::arg("onnx_path"), py::arg("use_cuda") = false)
        .def_property_readonly("type", &VolumeFieldPredictor::type)
        .def_property_readonly("is_voxelwise", &VolumeFieldPredictor::is_voxelwise)
        .def_property_readonly("spectrum_bins", &VolumeFieldPredictor::spectrum_bins)
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
             py::arg("beam"), py::arg("dims"), py::arg("max_inner_batch") = 65536);

    py::class_<VoxelFieldPredictor, VolumeFieldPredictor,
               std::shared_ptr<VoxelFieldPredictor>>(m, "VoxelFieldPredictor", py::dynamic_attr())
        .def("encode_beam", &VoxelFieldPredictor::encode_beam, py::arg("beam"))
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
             py::arg("positions"), py::arg("encoded_beam"));

    // ── RF3M container (rfnn::io::V1) ────────────────────────────────────────
    using rfnn::io::V1::ModelStore;

    // Read/WRITE + constructible: these double as the SAVE-side metadata builders, so the Python
    // packager assembles a domain/provenance here and the bytes are produced by the SAME C++
    // serialiser the deploy lib uses (no duplicated byte layout).
    py::class_<rfnn::io::ParameterRange>(m, "ParameterRange")
        .def(py::init([](float mn, float mx, std::string u) {
                 return rfnn::io::ParameterRange{mn, mx, std::move(u)};
             }), py::arg("min") = 0.f, py::arg("max") = 0.f, py::arg("unit") = "")
        .def_readwrite("min", &rfnn::io::ParameterRange::min)
        .def_readwrite("max", &rfnn::io::ParameterRange::max)
        .def_readwrite("unit", &rfnn::io::ParameterRange::unit);
    py::class_<rfnn::io::BeamParameter>(m, "BeamParameterSpec")
        .def(py::init([](std::string n, int c, rfnn::io::ParameterRange r) {
                 return rfnn::io::BeamParameter{std::move(n), c, std::move(r)};
             }), py::arg("name"), py::arg("count"), py::arg("range") = rfnn::io::ParameterRange{})
        .def_readwrite("name", &rfnn::io::BeamParameter::name)
        .def_readwrite("count", &rfnn::io::BeamParameter::count)
        .def_readwrite("range", &rfnn::io::BeamParameter::range);
    py::class_<rfnn::io::ModelDomain>(m, "ModelDomain")
        .def(py::init([](int bins, float max_e, std::vector<rfnn::io::BeamParameter> bp) {
                 rfnn::io::ModelDomain d; d.spectrum_bins = bins; d.spectrum_max_energy_ev = max_e;
                 d.beam_parameters = std::move(bp); return d;
             }), py::arg("spectrum_bins") = 0, py::arg("spectrum_max_energy_ev") = 0.f,
             py::arg("beam_parameters") = std::vector<rfnn::io::BeamParameter>{})
        .def_readwrite("spectrum_bins", &rfnn::io::ModelDomain::spectrum_bins)
        .def_readwrite("spectrum_max_energy_ev", &rfnn::io::ModelDomain::spectrum_max_energy_ev)
        .def_readwrite("beam_parameters", &rfnn::io::ModelDomain::beam_parameters);
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
        .def_static("load_from_memory",
                    [](py::bytes data, bool use_cuda)
                        -> std::shared_ptr<VolumeFieldPredictor> {
                        std::string s = data;
                        return ModelStore::load_from_memory(s.data(), s.size(), use_cuda);
                    },
                    py::arg("data"), py::arg("use_cuda") = false);

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
    m.def("save_to_memory",
          [to_named_graphs](const py::dict& graphs, const rfnn::io::ModelDomain& domain,
                            const rfnn::io::ModelProvenance& prov,
                            const std::map<std::string, float>& metrics) {
              auto bytes = ModelStore::save_to_memory(to_named_graphs(graphs), domain, prov, metrics);
              return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          },
          py::arg("graphs"), py::arg("domain"), py::arg("provenance"), py::arg("metrics"),
          "Serialise an RF3M package to bytes (named ONNX graphs + domain + provenance + metrics).");
    m.def("save",
          [to_named_graphs](const std::string& path, const py::dict& graphs,
                            const rfnn::io::ModelDomain& domain, const rfnn::io::ModelProvenance& prov,
                            const std::map<std::string, float>& metrics) {
              ModelStore::save(path, to_named_graphs(graphs), domain, prov, metrics);
          },
          py::arg("path"), py::arg("graphs"), py::arg("domain"), py::arg("provenance"), py::arg("metrics"),
          "Write an RF3M package straight to disk.");
}
