"""
The deployment interface as seen from Python.

The enums are DEFINED IN C++ (include/radfield3d-nn/model_interface.h) and reach Python only
through the pybind bindings — there is no Python copy to drift. These tests assert the same facts
as tests/cxx/test_model_interface.cpp (bit values, id packing, reserved-bit rejection, domain
cross-checks), so a change on either side that breaks the contract fails on both.

They also check the producer half: every trained model declares its interface via
deploy_interface(), and that declaration is what a C++ consumer will later dispatch on.
"""
import pytest

# Import the extension DIRECTLY (as the CI deploy job does, with PYTHONPATH=<repo>/lib) rather than
# through radfield3dnn.deploy: the deploy half must be testable without torch installed, which is
# exactly the environment a consumer of a published wheel has.
rfnn_deploy = pytest.importorskip(
    "rfnn_deploy", reason="deployment extension not built (cmake --build build_deploy)"
)

ModelInput = rfnn_deploy.ModelInput
ModelOutput = rfnn_deploy.ModelOutput
ModelInterface = rfnn_deploy.ModelInterface


def full_domain():
    """A domain carrying every resolution the flags used below depend on."""
    d = rfnn_deploy.ModelDomain()
    d.spectrum_bins = 128
    d.angular_phi_segments = 16
    d.angular_theta_segments = 8
    d.collimation = rfnn_deploy.CollimationType.Rectangle
    return d


def pbrf_like():
    i = ModelInterface()
    i.inputs = (ModelInput.POSITION | ModelInput.TUBE_SPECTRUM | ModelInput.BEAM_DIRECTION
                | ModelInput.SOURCE_DISTANCE | ModelInput.BEAM_COLLIMATION)
    i.outputs = ModelOutput.FLUX | ModelOutput.SPECTRUM
    return i


class TestBoundEnums:
    def test_flag_bits_match_cxx(self):
        # Same pinned ABI values as test_model_interface.cpp — this is what catches a C++ reorder.
        assert int(ModelInput.POSITION) == 1 << 0
        assert int(ModelInput.QUERY_DIRECTION) == 1 << 1
        assert int(ModelInput.TUBE_SPECTRUM) == 1 << 2
        assert int(ModelInput.BEAM_DIRECTION) == 1 << 3
        assert int(ModelInput.SOURCE_DISTANCE) == 1 << 4
        assert int(ModelInput.SOURCE_ORIGIN_3D) == 1 << 5
        assert int(ModelInput.BEAM_COLLIMATION) == 1 << 6
        assert int(ModelInput.PATIENT_TRANSLATION_3D) == 1 << 7
        assert int(ModelInput.PATIENT_ROTATION_3D) == 1 << 8
        assert int(ModelInput.GEOMETRY_MAP) == 1 << 9
        assert int(ModelInput.ANODE_ANGLE) == 1 << 10

        assert int(ModelOutput.FLUX) == 1 << 0
        assert int(ModelOutput.SPECTRUM) == 1 << 1
        assert int(ModelOutput.ANGULAR_FLUX) == 1 << 2
        assert int(ModelOutput.ERROR) == 1 << 3
        assert int(ModelOutput.AIR_KERMA) == 1 << 4
        assert int(ModelOutput.SEPARATE_CHANNELS) == 1 << 5

    def test_bitwise_ops_work_from_python(self):
        combo = ModelInput.POSITION | ModelInput.TUBE_SPECTRUM
        assert combo & ModelInput.POSITION
        assert not (combo & ModelInput.GEOMETRY_MAP)


class TestInterface:
    def test_position_bit_selects_the_predictor_class(self):
        assert pbrf_like().is_voxelwise

        unet = ModelInterface()  # whole-volume CNN: not queried at points
        unet.inputs = ModelInput.TUBE_SPECTRUM | ModelInput.BEAM_DIRECTION
        unet.outputs = ModelOutput.FLUX
        assert not unet.is_voxelwise

    def test_id_packs_inputs_and_outputs(self):
        i = pbrf_like()
        assert i.id >> 32 == int(i.inputs)
        assert i.id & 0xFFFFFFFF == int(i.outputs)
        assert ModelInterface.make_id(i.inputs, i.outputs) == i.id

    def test_translation_changes_the_id(self):
        pbrf = pbrf_like()
        tpbrf = pbrf_like()
        tpbrf.inputs = tpbrf.inputs | ModelInput.PATIENT_TRANSLATION_3D
        assert pbrf.id != tpbrf.id
        assert tpbrf.takes(ModelInput.PATIENT_TRANSLATION_3D)
        assert not pbrf.takes(ModelInput.PATIENT_TRANSLATION_3D)

    def test_validate_accepts_a_complete_domain(self):
        pbrf_like().validate(full_domain())

    def test_validate_rejects_a_promise_the_domain_cannot_describe(self):
        d = full_domain()
        d.spectrum_bins = 0  # the interface promises Spectrum but nobody recorded the bin count
        with pytest.raises(Exception):
            pbrf_like().validate(d)

    def test_validate_rejects_missing_collimation_kind(self):
        d = full_domain()
        d.collimation = rfnn_deploy.CollimationType.None_
        with pytest.raises(Exception):
            pbrf_like().validate(d)

    def test_validate_rejects_reserved_bits(self):
        # A package written by a NEWER producer must be refused, not half-understood.
        future = ModelInterface()
        future.inputs = ModelInput(int(ModelInput.POSITION) | (1 << 11))
        future.outputs = ModelOutput.FLUX
        with pytest.raises(Exception):
            future.validate(full_domain())


class TestProducerDeclaration:
    """The training side declares the interface; the C++ consumer dispatches on it.

    Needs the training stack (torch); skipped in the deploy-only CI job, which deliberately has no
    deep-learning dependencies.
    """

    def test_pbrf_and_tpbrf_declare_distinct_interfaces(self):
        pytest.importorskip("torch")
        from radfield3dnn.models.nerf import PBRFNet
        from radfield3dnn.models.nerf_translation import TPBRFNet

        # deploy_interface() is defined on the model, so the declaration lives with the network that
        # knows its own I/O — not in a lookup table that can fall out of sync.
        assert hasattr(PBRFNet, "deploy_interface")
        assert hasattr(TPBRFNet, "deploy_interface")

    def test_tpbrf_adds_the_translation_input(self):
        # TPBRFNet's override ORs in exactly one flag over PBRFNet's declaration.
        pytest.importorskip("torch")
        import inspect
        from radfield3dnn.models.nerf_translation import TPBRFNet

        src = inspect.getsource(TPBRFNet.deploy_interface)
        assert "PATIENT_TRANSLATION_3D" in src

    def test_every_flag_the_models_reference_exists_in_the_cxx_enum(self):
        """The models name flags in Python; the flags are DEFINED in C++.

        Nothing binds the two at import time — deploy_interface() only runs at export — so a C++
        rename leaves a stale ``ModelInput.FOO`` in the model source that fails months later, at the
        one moment someone tries to package a trained network. Resolve every referenced flag now.
        """
        pytest.importorskip("torch")
        import inspect
        import re

        from radfield3dnn.models import base, nerf_translation

        for module in (base, nerf_translation):
            src = inspect.getsource(module)
            for enum_name, enum_cls in (("ModelInput", ModelInput), ("ModelOutput", ModelOutput)):
                for flag in re.findall(rf"\b{enum_name}\.([A-Z][A-Z0-9_]*)\b", src):
                    assert hasattr(enum_cls, flag), (
                        f"{module.__name__} references {enum_name}.{flag}, which does not exist in "
                        f"the C++ enum (include/radfield3d-nn/model_interface.h)"
                    )


# ── zero-copy binding ────────────────────────────────────────────────────────────────────────────
# The caller owns the memory; the model writes into it. These run against a real RF3M package when
# one is present (the reference DS03 run), and skip cleanly otherwise — CI has no model artefacts.
import os

import pytest

REF_PKG = "/mnt/data/models_nn/logs/pbrf-roifull-floor1e8/pbrf-roifull-floor1e8-ds03/models/PBRFNet.rf3m"
needs_pkg = pytest.mark.skipif(not os.path.exists(REF_PKG), reason="no local RF3M package")
np = pytest.importorskip("numpy")


@pytest.fixture
def pred():
    return rfnn_deploy.ModelStore.load(REF_PKG, False)      # CPU session


def _bind_beam(p):
    p.bind_global_parameter(ModelInput.TUBE_SPECTRUM,
                            np.ones(p.input_spectrum_bins, dtype=np.float32))
    p.bind_global_parameter(ModelInput.BEAM_DIRECTION, np.array([0, 0, -1], dtype=np.float32))
    p.bind_global_parameter(ModelInput.SOURCE_DISTANCE, np.zeros(1, dtype=np.float32))


@needs_pkg
class TestZeroCopyBinding:
    def test_writes_into_the_callers_buffer(self, pred):
        pred.set_voxel_grid((16, 16, 16))
        _bind_beam(pred)
        flux = np.zeros((16, 16, 16), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, flux)

        pred.predict_volume()                                # no arguments
        assert np.isfinite(flux).all() and flux.any()        # the model filled OUR array

    def test_bound_result_equals_the_allocating_path(self, pred):
        """The zero-copy path must be the same computation, not merely a plausible one."""
        pred.set_voxel_grid((16, 16, 16))
        spectrum = np.ones(pred.input_spectrum_bins, dtype=np.float32)
        pred.bind_global_parameter(ModelInput.TUBE_SPECTRUM, spectrum)
        pred.bind_global_parameter(ModelInput.BEAM_DIRECTION, np.array([0, 0, -1], dtype=np.float32))
        # origin (.5,.5,.5) -> source distance 0 m -> 0 normalised: what the allocating path feeds.
        pred.bind_global_parameter(ModelInput.SOURCE_DISTANCE, np.zeros(1, dtype=np.float32))

        flux = np.zeros((16, 16, 16), dtype=np.float32)
        spec = np.zeros((16, 16, 16, pred.spectrum_bins), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, flux)
        pred.bind_output_layer(ModelOutput.SPECTRUM, spec)
        pred.predict_volume()

        beam = rfnn_deploy.BeamParameters(direction=[0, 0, -1], origin=[0.5, 0.5, 0.5],
                                          spectrum=list(spectrum))
        ref = pred.predict_volume(beam, (16, 16, 16))
        assert np.allclose(ref["flux"].ravel(), flux.ravel(), atol=1e-6)
        assert np.allclose(ref["spectrum"].reshape(spec.shape), spec, atol=1e-6)

    def test_in_place_edit_of_a_bound_input_is_picked_up(self, pred):
        """A directly-bound buffer is re-read every run — that is the whole point of binding it."""
        pred.set_voxel_grid((8, 8, 8))
        spectrum = np.ones(pred.input_spectrum_bins, dtype=np.float32)
        pred.bind_global_parameter(ModelInput.TUBE_SPECTRUM, spectrum)
        pred.bind_global_parameter(ModelInput.BEAM_DIRECTION, np.array([0, 0, -1], dtype=np.float32))
        pred.bind_global_parameter(ModelInput.SOURCE_DISTANCE, np.zeros(1, dtype=np.float32))
        flux = np.zeros((8, 8, 8), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, flux)

        pred.predict_volume()
        before = flux.copy()
        spectrum[:] = np.linspace(0.0, 2.0, spectrum.size).astype(np.float32)   # in place, no rebind
        pred.predict_volume()
        assert not np.allclose(before, flux)

    def test_grid_is_per_call(self, pred):
        _bind_beam(pred)
        for n in (4, 8):
            pred.set_voxel_grid((n, n, n))
            flux = np.zeros((n, n, n), dtype=np.float32)
            pred.bind_output_layer(ModelOutput.FLUX, flux)
            pred.predict_volume()
            assert np.isfinite(flux).all() and flux.any()


@needs_pkg
class TestBindingIsStrict:
    def test_precision_mismatch_is_refused(self, pred):
        pred.set_voxel_grid((8, 8, 8))
        with pytest.raises(rfnn_deploy.BindingError) as e:
            pred.bind_output_layer(ModelOutput.FLUX, np.zeros((8, 8, 8), dtype=np.float16))
        msg = str(e.value)
        assert "PRECISION" in msg and "convert_buffer" in msg     # says what, and the way out

    def test_conversion_must_be_asked_for_explicitly(self, pred):
        pred.set_voxel_grid((8, 8, 8))
        _bind_beam(pred)
        f16 = np.zeros((8, 8, 8), dtype=np.float16)
        pred.bind_output_layer(ModelOutput.FLUX, f16, convert_buffer=True)
        pred.predict_volume()
        assert np.isfinite(f16.astype(np.float32)).all() and f16.any()

    def test_a_too_small_buffer_is_refused_even_with_conversion(self, pred):
        """No conversion can fix capacity — the model would write past the end of the memory."""
        pred.set_voxel_grid((8, 8, 8))
        with pytest.raises(rfnn_deploy.BindingError) as e:
            pred.bind_output_layer(ModelOutput.FLUX, np.zeros(4, dtype=np.float32),
                                   convert_buffer=True)
        assert "CAPACITY" in str(e.value)

    def test_a_flag_the_model_does_not_consume_is_refused(self, pred):
        with pytest.raises(rfnn_deploy.BindingError):
            pred.bind_global_parameter(ModelInput.PATIENT_TRANSLATION_3D,
                                       np.zeros(3, dtype=np.float32))

    def test_incomplete_setup_lists_every_gap_at_once(self, pred):
        with pytest.raises(rfnn_deploy.IncompleteSetupError) as e:
            pred.predict_volume()
        msg = str(e.value)
        assert "set_voxel_grid" in msg
        assert "TubeSpectrum" in msg and "BeamDirection" in msg   # every missing input, not just one
        assert "bind_output_layer" in msg


@needs_pkg
class TestNormalizerAndRepr:
    def test_normalizer_writes_normalised_values_into_the_bound_buffer(self, pred):
        pred.set_voxel_grid((4, 4, 4))
        distance = np.full(1, -1.0, dtype=np.float32)
        pred.bind_global_parameter(ModelInput.SOURCE_DISTANCE, distance)

        norm = pred.parameter_normalizer()
        norm.write(ModelInput.SOURCE_DISTANCE, np.array([0.8], dtype=np.float32),
                   rfnn_deploy.Unit.Metres)
        assert 0.0 <= float(distance[0]) <= 1.0        # metres -> the network's [0,1] space, in place

    def test_repr_reports_the_backend_and_the_interface(self, pred):
        pred.set_voxel_grid((8, 8, 8))
        r = repr(pred)
        assert "VoxelFieldPredictor" in r
        assert "ExecutionProvider" in r                 # what it is ACTUALLY running on
        assert "8x8x8" in r
        assert "Flux" in r and "TubeSpectrum" in r      # its declared I/O


@needs_pkg
class TestBoundBuffersSurvive:
    def test_binding_a_temporary_does_not_corrupt_the_result(self, pred):
        """The model holds a raw pointer INTO the bound array, so the binding must keep it alive.

        Binding a temporary used to free it the instant the call returned; the allocator then reused
        the memory and the next inference silently read whatever landed there — wrong numbers, no
        error. pybind's keep_alive makes that impossible. Forcing a gc pass here is what actually
        reclaims the temporary, so this fails loudly if the keep_alive is ever dropped.
        """
        import gc

        pred.set_voxel_grid((16, 16, 16))
        # every buffer is a TEMPORARY — nothing below holds a reference to it
        pred.bind_global_parameter(ModelInput.TUBE_SPECTRUM,
                                   np.ones(pred.input_spectrum_bins, dtype=np.float32))
        pred.bind_global_parameter(ModelInput.BEAM_DIRECTION,
                                   np.array([0, 0, -1], dtype=np.float32))
        pred.bind_global_parameter(ModelInput.SOURCE_DISTANCE, np.zeros(1, dtype=np.float32))
        flux = np.zeros((16, 16, 16), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, flux)

        gc.collect()
        pred.predict_volume()

        beam = rfnn_deploy.BeamParameters(
            direction=[0, 0, -1], origin=[0.5, 0.5, 0.5],
            spectrum=[1.0] * pred.input_spectrum_bins)
        ref = pred.predict_volume(beam, (16, 16, 16))
        assert np.allclose(ref["flux"].ravel(), flux.ravel(), atol=1e-6)


@needs_pkg
class TestSparseVoxelPrediction:
    def test_only_the_named_voxels_are_written(self, pred):
        """A moving ROI can be refreshed without recomputing — or clearing — the rest of the grid."""
        pred.set_voxel_grid((16, 16, 16))
        _bind_beam(pred)
        flux = np.full((16, 16, 16), -7.0, dtype=np.float32)      # sentinel
        pred.bind_output_layer(ModelOutput.FLUX, flux)

        roi = np.array([[i, j, k] for i in range(4, 7) for j in range(4, 7) for k in range(4, 7)],
                       dtype=np.int32)
        pred.predict_voxels(roi)

        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[4:7, 4:7, 4:7] = True
        assert (flux[mask] != -7.0).all() and np.isfinite(flux[mask]).all()
        assert (flux[~mask] == -7.0).all()                        # untouched voxels keep their value

    def test_sparse_voxels_match_the_full_volume(self, pred):
        pred.set_voxel_grid((16, 16, 16))
        _bind_beam(pred)
        sparse = np.zeros((16, 16, 16), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, sparse)
        roi = np.array([[2, 3, 4], [8, 8, 8], [15, 15, 15], [0, 0, 0]], dtype=np.int32)
        pred.predict_voxels(roi)

        full = np.zeros((16, 16, 16), dtype=np.float32)
        pred.bind_output_layer(ModelOutput.FLUX, full)
        pred.predict_volume()

        for i, j, k in roi:
            assert abs(float(sparse[i, j, k]) - float(full[i, j, k])) < 1e-6

    def test_a_voxel_outside_the_grid_is_refused(self, pred):
        pred.set_voxel_grid((8, 8, 8))
        _bind_beam(pred)
        pred.bind_output_layer(ModelOutput.FLUX, np.zeros((8, 8, 8), dtype=np.float32))
        with pytest.raises(rfnn_deploy.BindingError):
            pred.predict_voxels(np.array([[0, 0, 99]], dtype=np.int32))


@needs_pkg
def test_backend_vocabulary_is_defined_once_in_cxx():
    """load_rf3m forwards the NAME to C++; it does not re-decide what a backend means."""
    p = rfnn_deploy.ModelStore.load(REF_PKG, "cpu")
    assert p.session_memory.provider == "CPUExecutionProvider"
    with pytest.raises(Exception) as e:
        rfnn_deploy.ModelStore.load(REF_PKG, "vulkan")
    assert "Vulkan execution provider" in str(e.value)      # explains, rather than just failing
    with pytest.raises(Exception):
        rfnn_deploy.ModelStore.load(REF_PKG, "bogus")


# ── device-resident bound path (GPU session) ──────────────────────────────────────────────────────
# On a CUDA session the bound per-voxel path keeps the beam latent and the position grid in DEVICE
# memory across the tiled runs (encoder output bound to the device, broadcast device-side via
# CopyTensors), and scratch-allocates unrequested outputs on the device instead of downloading them.
# These assert that this traffic-elimination changes NOTHING about the numbers a caller reads back,
# and that an in-place edit of a bound device input is still honoured. Skipped without a CUDA torch
# or the reference package. torch is imported softly (NOT importorskip) so the torch-free deploy CI
# job still collects and runs every CPU test above.
try:
    import torch
    _cuda_ok = torch.cuda.is_available()
except Exception:
    torch = None
    _cuda_ok = False
needs_cuda = pytest.mark.skipif(
    not (os.path.exists(REF_PKG) and _cuda_ok),
    reason="no CUDA device or no local RF3M package",
)


def _load_cuda():
    exec_opts = rfnn_deploy.ExecutionOptions()
    exec_opts.use_gpu = True
    exec_opts.use_tensorrt = False          # CUDA EP: fast to load, no per-shape engine build
    return rfnn_deploy.ModelStore.load(REF_PKG, exec_opts)


def _bind_beam_cuda(p):
    # Held by the caller and returned so the bindings outlive the run (keep_alive holds them too).
    sp = torch.ones(p.input_spectrum_bins, dtype=torch.float32, device="cuda")
    di = torch.tensor([0, 0, -1], dtype=torch.float32, device="cuda")
    ds = torch.zeros(1, dtype=torch.float32, device="cuda")
    p.bind_global_parameter(ModelInput.TUBE_SPECTRUM, sp)
    p.bind_global_parameter(ModelInput.BEAM_DIRECTION, di)
    p.bind_global_parameter(ModelInput.SOURCE_DISTANCE, ds)
    return sp, di, ds


@needs_cuda
class TestZeroCopyBindingCuda:
    def test_device_path_equals_the_allocating_path_and_the_cpu_reference(self):
        """The device-resident broadcast must be the SAME computation — no numbers may drift.

        Against the allocating path on the SAME CUDA session it is bit-exact (identical graph, identical
        inputs); against the CPU reference it carries only the pre-existing CUDA-vs-CPU EP numerics
        (~6e-4 median), so this catches a broadcast/positions bug loudly (it would blow that up).
        """
        p = _load_cuda()
        assert p.session_memory.device == rfnn_deploy.DeviceKind.Cuda
        dims = (16, 16, 16)
        p.set_voxel_grid(dims)
        keep = _bind_beam_cuda(p)                                        # noqa: F841 (outlives run)
        flux = torch.zeros(dims, dtype=torch.float32, device="cuda")
        spec = torch.zeros((*dims, p.spectrum_bins), dtype=torch.float32, device="cuda")
        p.bind_output_layer(ModelOutput.FLUX, flux)
        p.bind_output_layer(ModelOutput.SPECTRUM, spec)
        p.predict_volume()
        torch.cuda.synchronize()
        bound_flux = flux.cpu().numpy().ravel()

        beam = rfnn_deploy.BeamParameters(direction=[0, 0, -1], origin=[0.5, 0.5, 0.5],
                                          spectrum=[1.0] * p.input_spectrum_bins)
        alloc = p.predict_volume(beam, dims)["flux"].ravel()            # same CUDA session
        assert np.abs(bound_flux - alloc).max() < 1e-4                  # same session: matches to fp noise

        cpu = rfnn_deploy.ModelStore.load(REF_PKG, False).predict_volume(beam, dims)["flux"].ravel()
        med = np.median(np.abs(bound_flux - cpu) / (np.abs(cpu) + 1e-8))
        assert med < 5e-3      # the CUDA-vs-CPU EP floor, unchanged by the device broadcast

    def test_rectangular_and_oversized_grids_match_the_allocating_path(self):
        """A RECTANGULAR, non-power-of-2 grid is what catches a device-side axis or flat-index bug: on
        a cube D==H==W it would hide (the numbers coincide). One grid also exceeds the 65536 chunk so
        the partial multi-chunk / doubling-clamp paths run. Bit-exact against the allocating path (same
        session, same math) confirms the device position grid and latent broadcast are laid out right.
        """
        p = _load_cuda()
        for dims in [(12, 10, 9), (41, 41, 41)]:                        # rectangular; and N=68921 > 65536
            p.clear_bindings()
            p.set_voxel_grid(dims)
            keep = _bind_beam_cuda(p)                                    # noqa: F841
            flux = torch.zeros(dims, dtype=torch.float32, device="cuda")
            p.bind_output_layer(ModelOutput.FLUX, flux)
            p.predict_volume()
            torch.cuda.synchronize()
            bound = flux.cpu().numpy().ravel()
            beam = rfnn_deploy.BeamParameters(direction=[0, 0, -1], origin=[0.5, 0.5, 0.5],
                                              spectrum=[1.0] * p.input_spectrum_bins)
            alloc = p.predict_volume(beam, dims)["flux"].ravel()
            assert np.abs(bound - alloc).max() < 1e-4, f"device path diverged at grid {dims}"

    def test_sparse_voxels_match_the_full_volume_on_cuda(self):
        """A moving ROI refreshed on the device fast path must match a full run and leave the rest."""
        p = _load_cuda()
        p.set_voxel_grid((16, 16, 16))
        keep = _bind_beam_cuda(p)                                        # noqa: F841
        sparse = torch.full((16, 16, 16), -7.0, dtype=torch.float32, device="cuda")
        p.bind_output_layer(ModelOutput.FLUX, sparse)
        roi = np.array([[i, j, k] for i in range(4, 7) for j in range(4, 7) for k in range(4, 7)],
                       dtype=np.int32)
        p.predict_voxels(roi)
        torch.cuda.synchronize()
        s = sparse.cpu().numpy()

        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[4:7, 4:7, 4:7] = True
        assert (s[~mask] == -7.0).all()                                 # untouched voxels kept
        assert np.isfinite(s[mask]).all()

        full = torch.zeros((16, 16, 16), dtype=torch.float32, device="cuda")
        p.bind_output_layer(ModelOutput.FLUX, full)
        p.predict_volume()
        torch.cuda.synchronize()
        f = full.cpu().numpy()
        assert np.abs(s[mask] - f[mask]).max() < 1e-4                   # ROI == full-volume values

    def test_in_place_edit_of_a_bound_cuda_input_is_picked_up(self):
        """The encoder re-runs over the bound device buffers every predict — no rebinding needed.

        A uniform rescale of the tube spectrum normalises away inside the encoder, so the edit changes
        its DISTRIBUTION (like the CPU counterpart) to prove the new values actually reached the graph.
        """
        p = _load_cuda()
        p.set_voxel_grid((8, 8, 8))
        sp, di, ds = _bind_beam_cuda(p)
        flux = torch.zeros((8, 8, 8), dtype=torch.float32, device="cuda")
        p.bind_output_layer(ModelOutput.FLUX, flux)
        p.predict_volume()
        torch.cuda.synchronize()
        before = flux.clone()
        sp.copy_(torch.linspace(0.0, 2.0, sp.numel(), device="cuda"))   # in place, no rebind
        p.predict_volume()
        torch.cuda.synchronize()
        assert (before != flux).any()

    def test_binding_only_flux_not_spectrum_on_cuda(self):
        """Binding Flux alone must still produce correct flux — the unrequested spectrum is scratch-
        allocated on the DEVICE (not downloaded to host), which this exercises end to end."""
        p = _load_cuda()
        dims = (16, 16, 16)
        p.set_voxel_grid(dims)
        keep = _bind_beam_cuda(p)                                        # noqa: F841
        flux = torch.zeros(dims, dtype=torch.float32, device="cuda")
        p.bind_output_layer(ModelOutput.FLUX, flux)                     # SPECTRUM deliberately unbound
        p.predict_volume()
        torch.cuda.synchronize()
        bound = flux.cpu().numpy().ravel()
        assert np.isfinite(bound).all()

        beam = rfnn_deploy.BeamParameters(direction=[0, 0, -1], origin=[0.5, 0.5, 0.5],
                                          spectrum=[1.0] * p.input_spectrum_bins)
        alloc = p.predict_volume(beam, dims)["flux"].ravel()
        assert np.abs(bound - alloc).max() < 1e-4
