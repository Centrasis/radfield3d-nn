#include "gtest/gtest.h"
#include "radfield3d-nn/tcnn/base_model.h"
#include "radfield3d-nn/tcnn/layers/gated_fusion.h"
#include "radfield3d-nn/tcnn/layers/beam_fusion.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <chrono>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
using namespace std::chrono;

namespace {
	TEST(LocationEncoding, Creation) {
		cudaFree(0); 

		int deviceID;
		cudaError_t err = cudaGetDevice(&deviceID);

		if (err == cudaSuccess) {
			std::cout << "Currently active device ID: " << deviceID << std::endl;
		} else {
			std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		}
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, deviceID);

		std::cout << "Device Name: " << prop.name << std::endl;
		std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

		auto encoding = std::make_shared<rfnn::tcnn::LocationEncoding>(12, 192);
	}

	TEST(LocationEncoding, Inference) {
		cudaFree(0); 
		size_t resolution = 64;
		auto locations = ::tcnn::GPUMatrixDynamic<float>(3, resolution * resolution * resolution);
		auto encoding = std::make_shared<rfnn::tcnn::LocationEncoding>(12, 192);
		auto weights = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(encoding->n_params(), 1);
		auto grads = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(encoding->n_params(), 1);
		encoding->set_params(weights.data(), weights.data(), grads.data());

		auto encoded = encoding->encode_locations(locations);
		assert(encoded->m() == 192);
	}

	TEST(LocationEncoding, Backward) {
		cudaFree(0); 
		size_t resolution = 64;
		auto locations = ::tcnn::GPUMatrixDynamic<float>(3, resolution * resolution * resolution);
		auto encoding = std::make_shared<rfnn::tcnn::LocationEncoding>(12, 192);
		auto encoded_locations = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(192, resolution * resolution * resolution);
		auto loss = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(192, resolution * resolution * resolution);
		auto weights = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(encoding->n_params(), 1);
		auto grads = ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(encoding->n_params(), 1);
		auto grads_input = ::tcnn::GPUMatrixDynamic<float>(3, resolution * resolution * resolution);
		encoding->set_params(weights.data(), weights.data(), grads.data());

		auto ctx = encoding->forward(
			locations,
			&encoded_locations,
			false,
			true
		);

		cudaStreamSynchronize(0);

		encoding->backward(
			*ctx,
			locations,
			encoded_locations,
			loss,
			&grads_input
		);
	}

	// BaseRadiationPredictionModel ctor signature (current):
	//   (d_model, location_encoding_dim, flux_offset, flux_activation,
	//    location_encoding_kind, flux_clamp_min, flux_clamp_max).
	TEST(Network, Creation) {
		cudaFree(0);
		// d_model=128, location_encoding_dim=12, flux_offset=0.5 (defaults
		// for everything else). Smoke-test the ctor wires the flux
		// projector and the param block sums to the expected width.
		auto network = std::make_shared<rfnn::tcnn::BaseRadiationPredictionModel>(128u, 12u);
		ASSERT_GT(network->n_params(), 0u);
		// Single-head output: flux (1) + spectrum (32) = 33.
		ASSERT_EQ(network->output_width(), 33u);
		ASSERT_EQ(network->padded_output_width(), 33u);
		// Input width: xyz (3) + d_model beam features.
		ASSERT_EQ(network->input_width(), 3u + 128u);
	}

	TEST(Network, ClampsAreReportedInHyperparams) {
		cudaFree(0);
		auto network = std::make_shared<rfnn::tcnn::BaseRadiationPredictionModel>(
			/*d_model=*/64u, /*location_encoding_dim=*/12u,
			/*flux_offset=*/0.3f, /*flux_activation=*/0,
			rfnn::tcnn::LocationEncodingKind::Frequency,
			/*flux_clamp_min=*/0.0f,    /*flux_clamp_max=*/1.0f
		);
		auto hp = network->hyperparams();
		ASSERT_EQ(hp.at("otype").get<std::string>(), "BaseRadiationPredictionModel");
		ASSERT_EQ(hp.at("flux_offset").get<float>(), 0.3f);
		ASSERT_EQ(hp.at("flux_clamp_min").get<float>(), 0.0f);
		ASSERT_EQ(hp.at("flux_clamp_max").get<float>(), 1.0f);
	}

	TEST(Network, JitDeviceFunctionGenerationCompiles) {
		// The real model has no host forward — every call path goes through
		// the JIT-fused device functions produced by ``generate_device_function``
		// / ``generate_backward_device_function``. The strongest sanity test
		// we can run from the C++ side is that those generators produce
		// non-empty source for the pipeline; the actual NVRTC compilation
		// happens at runtime through the python_api bridge and is covered by
		// ``tests/test_pbrfnet_cpp.py``.
		cudaFree(0);
		auto network = std::make_shared<rfnn::tcnn::BaseRadiationPredictionModel>(64u, 12u);
		const std::string fwd = network->generate_device_function("rf_fwd");
		const std::string bwd = network->generate_backward_device_function("rf_bwd", 128u);
		ASSERT_FALSE(fwd.empty());
		ASSERT_FALSE(bwd.empty());
		// Flux head device function must appear in the generated source.
		ASSERT_NE(fwd.find("rf_fwd_flu("),    std::string::npos);
		// Spectrum head's softplus + sum-norm path stays.
		ASSERT_NE(fwd.find("rf_fwd_spec("),   std::string::npos);
		// Backward emits the flux backward chain.
		ASSERT_NE(bwd.find("rf_bwd_flu_bwd("),    std::string::npos);
	}

	// ── GatedFusion (bounded hardsigmoid beam fusion) ──────────────────────
	TEST(GatedFusion, StructuralPropertiesMatchFiLM) {
		cudaFree(0);
		// A GatedFusion must be a drop-in for FiLM: same input/output widths
		// and the same internal param count (condition -> 2*F MLP), so swapping
		// it into BaseRadiationPredictionModel leaves the param offsets intact.
		const uint32_t F = 128u, C = 128u;
		auto gated = std::make_shared<rfnn::tcnn::GatedFusion>(F, C, "SiLU");
		auto film  = std::make_shared<rfnn::tcnn::FiLM>(F, C, "SiLU");
		ASSERT_EQ(gated->output_width(), F);
		ASSERT_EQ(gated->padded_output_width(), F);
		ASSERT_EQ(gated->input_width(), F + C);
		ASSERT_GT(gated->n_params(), 0u);
		ASSERT_EQ(gated->n_params(), film->n_params());
		ASSERT_EQ(gated->device_function_fwd_ctx_bytes(), film->device_function_fwd_ctx_bytes());
	}

	TEST(GatedFusion, HyperparamsReportOtype) {
		cudaFree(0);
		auto gated = std::make_shared<rfnn::tcnn::GatedFusion>(64u, 64u, "SiLU");
		auto hp = gated->hyperparams();
		ASSERT_EQ(hp.at("otype").get<std::string>(), "GatedFusion");
		ASSERT_EQ(hp.at("feature_channels").get<uint32_t>(), 64u);
		ASSERT_EQ(hp.at("non_linearity").get<std::string>(), "SiLU");
	}

	TEST(GatedFusion, JitDeviceFunctionGenerationCompiles) {
		// As with the main model, the host forward throws "Use JIT!" — the
		// strongest C++-side check is that the forward/backward device-function
		// generators emit non-empty source that references the gate/candidate
		// MLP. Numeric forward+backward correctness is covered end-to-end by
		// tests/test_pbrfnet_cpp.py (beam_fusion="gated").
		cudaFree(0);
		auto gated = std::make_shared<rfnn::tcnn::GatedFusion>(64u, 64u, "SiLU");
		const std::string fwd = gated->generate_device_function("gf");
		const std::string bwd = gated->generate_backward_device_function("gf_bwd", 128u);
		ASSERT_FALSE(fwd.empty());
		ASSERT_FALSE(bwd.empty());
		ASSERT_NE(fwd.find("gf_mlp_gate_candidate("),        std::string::npos);
		ASSERT_NE(bwd.find("gf_bwd_mlp_gate_candidate_bwd("), std::string::npos);
	}

	TEST(Network, GatedFusionVariantBuildsAndJitCompiles) {
		// BaseRadiationPredictionModel must accept beam_fusion=GatedFusion and
		// still compose a complete JIT-fused forward/backward (the gated
		// conditioner's device functions fold into the model's).
		cudaFree(0);
		auto network = std::make_shared<rfnn::tcnn::BaseRadiationPredictionModel>(
			/*d_model=*/64u, /*location_encoding_dim=*/12u,
			/*flux_offset=*/0.5f, /*flux_activation=*/0,
			rfnn::tcnn::LocationEncodingKind::HashGrid,
			/*flux_clamp_min=*/-9.0f, /*flux_clamp_max=*/0.0f,
			/*trunk_hidden_layers=*/1u,
			rfnn::tcnn::BeamFusionKind::GatedFusion
		);
		ASSERT_GT(network->n_params(), 0u);
		ASSERT_EQ(network->hyperparams().at("beam_fusion").get<int>(),
		          static_cast<int>(rfnn::tcnn::BeamFusionKind::GatedFusion));
		const std::string fwd = network->generate_device_function("rf_fwd");
		const std::string bwd = network->generate_backward_device_function("rf_bwd", 128u);
		ASSERT_FALSE(fwd.empty());
		ASSERT_FALSE(bwd.empty());
	}
};