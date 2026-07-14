// The deployment interface (model_interface.h): the ABI between a stored model and any consumer.
// The contract is pinned here — bit values, reserved-bit rejection, domain cross-checks, and the
// fact that ModelInput::Position alone selects the executable predictor class.
// tests/test_deploy_bindings.py asserts the SAME facts through the pybind layer, so the enums
// (defined once, in C++) cannot drift from what Python sees.
#include <gtest/gtest.h>

#include <radfield3d-nn/model_binding.h>
#include <radfield3d-nn/model_interface.h>

using namespace radfield3dnn;
using rfnn::io::CollimationType;
using rfnn::io::ModelDomain;

namespace {

// A domain that carries every resolution the flags below depend on.
ModelDomain full_domain() {
    ModelDomain d;
    d.spectrum_bins = 128;
    d.angular_phi_segments = 16;
    d.angular_theta_segments = 8;
    d.collimation = CollimationType::Rectangle;
    return d;
}

ModelInterface pbrf_like() {
    ModelInterface i;
    i.inputs = ModelInput::Position | ModelInput::TubeSpectrum | ModelInput::BeamDirection
             | ModelInput::SourceDistance | ModelInput::BeamCollimation;
    i.outputs = ModelOutput::Flux | ModelOutput::Spectrum;
    return i;
}

}  // namespace

// ── the flags are ABI: these numbers may never change ────────────────────────────────────────────
TEST(ModelInterfaceTest, FlagBitsArePinned) {
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::Position), 1u << 0);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::QueryDirection), 1u << 1);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::TubeSpectrum), 1u << 2);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::BeamDirection), 1u << 3);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::SourceDistance), 1u << 4);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::SourceOrigin3D), 1u << 5);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::BeamCollimation), 1u << 6);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::PatientTranslation3D), 1u << 7);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::PatientRotation3D), 1u << 8);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::GeometryMap), 1u << 9);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelInput::AnodeAngle), 1u << 10);

    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::Flux), 1u << 0);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::Spectrum), 1u << 1);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::AngularFlux), 1u << 2);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::Error), 1u << 3);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::AirKerma), 1u << 4);
    EXPECT_EQ(static_cast<std::uint32_t>(ModelOutput::SeparateChannels), 1u << 5);
}

// The Position bit alone decides which executable class a package gets.
TEST(ModelInterfaceTest, PositionBitSelectsThePredictorClass) {
    EXPECT_TRUE(pbrf_like().is_voxelwise());

    ModelInterface unet;  // a whole-volume CNN is not queried at points
    unet.inputs = ModelInput::TubeSpectrum | ModelInput::BeamDirection;
    unet.outputs = ModelOutput::Flux;
    EXPECT_FALSE(unet.is_voxelwise());
}

// Two architectures with identical I/O are interchangeable; adding an input changes the id.
TEST(ModelInterfaceTest, IdPacksInputsAndOutputsAndSeparatesTPBRF) {
    const auto pbrf = pbrf_like();
    auto tpbrf = pbrf_like();
    tpbrf.inputs = tpbrf.inputs | ModelInput::PatientTranslation3D;

    EXPECT_NE(pbrf.id(), tpbrf.id());
    EXPECT_EQ(pbrf.id() >> 32, static_cast<std::uint32_t>(pbrf.inputs));
    EXPECT_EQ(pbrf.id() & 0xffffffffu, static_cast<std::uint32_t>(pbrf.outputs));
    EXPECT_EQ(ModelInterface::make_id(pbrf.inputs, pbrf.outputs), pbrf.id());
    EXPECT_TRUE(tpbrf.takes(ModelInput::PatientTranslation3D));
    EXPECT_FALSE(pbrf.takes(ModelInput::PatientTranslation3D));
}

// A package written by a NEWER producer must be rejected, not half-understood.
TEST(ModelInterfaceTest, ReservedBitsAreRejected) {
    const auto domain = full_domain();

    ModelInterface future = pbrf_like();
    future.inputs = static_cast<ModelInput>(static_cast<std::uint32_t>(future.inputs) | (1u << 11));
    EXPECT_THROW(future.validate(domain), UnsupportedInterfaceError);

    ModelInterface future_out = pbrf_like();
    future_out.outputs =
        static_cast<ModelOutput>(static_cast<std::uint32_t>(future_out.outputs) | (1u << 6));
    EXPECT_THROW(future_out.validate(domain), UnsupportedInterfaceError);
}

// validate() is the store/load gate: a flag whose resolution nobody recorded is a broken package.
TEST(ModelInterfaceTest, ValidateCrossChecksTheDomain) {
    EXPECT_NO_THROW(pbrf_like().validate(full_domain()));

    ModelDomain no_bins = full_domain();
    no_bins.spectrum_bins = 0;
    EXPECT_THROW(pbrf_like().validate(no_bins), UnsupportedInterfaceError);  // promises Spectrum

    ModelDomain no_collimation = full_domain();
    no_collimation.collimation = CollimationType::None;
    EXPECT_THROW(pbrf_like().validate(no_collimation), UnsupportedInterfaceError);

    ModelInterface angular = pbrf_like();
    angular.outputs = angular.outputs | ModelOutput::AngularFlux;
    ModelDomain no_angles = full_domain();
    no_angles.angular_phi_segments = 0;
    EXPECT_THROW(angular.validate(no_angles), UnsupportedInterfaceError);

    ModelInterface nothing = pbrf_like();
    nothing.outputs = ModelOutput::None;
    EXPECT_THROW(nothing.validate(full_domain()), UnsupportedInterfaceError);
}

// SourceDistance and SourceOrigin3D are two spellings of the same quantity.
TEST(ModelInterfaceTest, MutuallyExclusiveOriginFlagsAreRejected) {
    ModelInterface both = pbrf_like();
    both.inputs = both.inputs | ModelInput::SourceOrigin3D;
    EXPECT_THROW(both.validate(full_domain()), UnsupportedInterfaceError);
}

TEST(ModelInterfaceTest, ResolutionAwareRequiresRegionStateDims) {
    ModelInterface ipe = pbrf_like();
    ipe.resolution_aware = true;
    ipe.region_state_dims = 0;
    EXPECT_THROW(ipe.validate(full_domain()), UnsupportedInterfaceError);

    ipe.region_state_dims = 3;
    EXPECT_NO_THROW(ipe.validate(full_domain()));
}

// ── the strict-bind diagnostic ───────────────────────────────────────────────────────────────────
// A silent host<->device copy per inference destroys the entire point of binding caller memory, so
// the message must name what mismatched, what it costs, and how to opt in on purpose.
TEST(BindingDiagnosticTest, MismatchMessageNamesEveryProblemAndTheWayOut) {
    MemoryRef host_fp16{nullptr, 16, DeviceKind::Cpu, ElementType::F16, 0};
    SessionMemory cuda_fp32{DeviceKind::Cuda, ElementType::F32, 0, "CUDAExecutionProvider"};

    const auto msg = describe_binding_mismatch("Flux", host_fp16, cuda_fp32, /*required=*/1024);

    EXPECT_NE(msg.find("Flux"), std::string::npos);
    EXPECT_NE(msg.find("CAPACITY"), std::string::npos);   // room for 8 values, 1024 needed
    EXPECT_NE(msg.find("DEVICE"), std::string::npos);     // CPU buffer, CUDA session
    EXPECT_NE(msg.find("PRECISION"), std::string::npos);  // fp16 buffer, fp32 kernels
    EXPECT_NE(msg.find("CUDAExecutionProvider"), std::string::npos);
    EXPECT_NE(msg.find("convert_buffer"), std::string::npos);   // the explicit opt-in
}

// Capacity is an ELEMENT-count question: a float16 buffer holding the right number of values is big
// enough — it only needs converting. Reporting it as "too small" would send the caller the wrong way.
TEST(BindingDiagnosticTest, CapacityIsCountedInElementsNotBytes) {
    MemoryRef fp16_512{nullptr, 1024, DeviceKind::Cpu, ElementType::F16, 0};   // 512 values
    SessionMemory cpu_fp32{DeviceKind::Cpu, ElementType::F32, 0, "CPUExecutionProvider"};

    const auto msg = describe_binding_mismatch("Flux", fp16_512, cpu_fp32, /*required=*/512);
    EXPECT_EQ(msg.find("* CAPACITY"), std::string::npos);   // it HAS 512 values
    EXPECT_NE(msg.find("* PRECISION"), std::string::npos);  // it is just the wrong type
}

TEST(BindingDiagnosticTest, FlagNamesRoundTripToStrings) {
    EXPECT_EQ(to_string(ModelInput::PatientTranslation3D), "PatientTranslation3D");
    EXPECT_EQ(to_string(ModelOutput::AirKerma), "AirKerma");
    EXPECT_EQ(to_string(DeviceKind::Vulkan), "Vulkan");
    EXPECT_EQ(to_string(ElementType::F16), "float16");
}
