#include "gtest/gtest.h"
#include <typeinfo>
#include "radfield3d-nn-bindings/utils.h"
#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include "radfield3d-nn/tcnn/encodings/global_parameters.h"
#include "radfield3d-nn/tcnn/encodings/beam_encoder.h"
#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include "radfield3d-nn/tcnn/base_model.h"

//#include "radfield3d-nn/tcnn/base_model.h"
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <exception>

#include <radfield3d-nn-bindings/bridge.inl>

using LocationEncodingBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::LocationEncoding, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LocationEncoding, float>>;
using ParameterSetEncodingBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::ParameterSetEncoding, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::ParameterSetEncoding, float>>;
using FiLMBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::FiLM, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::FiLM, float>>;
using LayerNormBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::LayerNorm, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LayerNorm, float>>;
using PBRFBeamEncoderBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::PBRFBeamEncoder, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::PBRFBeamEncoder, float>>;
using ModelMainBridge = rfnn::tcnn::autograd::ModuleBridge<rfnn::tcnn::BaseRadiationPredictionModel, rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::BaseRadiationPredictionModel, float>>;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace {
    TEST(TorchSanity, BasicAutograd) {
		gpuErrchk( cudaDeviceSynchronize() );
        auto x = torch::randn({ 2, 3 }, torch::requires_grad());
        auto y = x.sum();
        y.backward();
		gpuErrchk( cudaDeviceSynchronize() );
    }

	TEST(LocationEncoding, BridgeCreation) {
		gpuErrchk( cudaDeviceSynchronize() );
		auto encoding = std::make_shared<LocationEncodingBridge>(12, 128);
	}

	TEST(LocationEncoding, BridgeInference) {
		auto encoding = std::make_shared<LocationEncodingBridge>(12, 64);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor locations = torch::rand({ 64 * 64, 3 }, opts);
		ASSERT_EQ(locations.size(0), 64 * 64);
		ASSERT_EQ(locations.size(1), 3);
		ASSERT_TRUE(locations.is_cuda());

		auto encoded = encoding->forward<torch::Tensor>(locations);
		ASSERT_EQ(encoded.size(0), 64 * 64);
	}

	TEST(LocationEncoding, BridgeAutograd) {
		auto encoding = std::make_shared<LocationEncodingBridge>(12, 64);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor locations = torch::rand({ 64 * 64, 3 }, opts);
		ASSERT_EQ(locations.size(0), 64 * 64);
		ASSERT_EQ(locations.size(1), 3);
		ASSERT_TRUE(locations.is_cuda());

		auto encoded = encoding->forward(locations);
		ASSERT_EQ(encoded.size(0), 64 * 64);

		auto grad_fn = encoded.grad_fn();
		ASSERT_TRUE(grad_fn.get() != nullptr && grad_fn.get() != NULL && grad_fn);
	}

	TEST(LocationEncoding, BridgeBackward) {
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		auto encoding = std::make_shared<LocationEncodingBridge>(12, 64);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor locations = torch::rand({ 64 * 64, 3 }, opts);
		ASSERT_EQ(locations.size(0), 64 * 64);
		ASSERT_EQ(locations.size(1), 3);
		ASSERT_TRUE(locations.is_cuda());

		auto encoded = encoding->forward(locations);
		ASSERT_EQ(encoded.size(0), 64 * 64);
		ASSERT_TRUE(encoded.grad_fn());

		torch::Tensor encoded_target = torch::rand_like(encoded);
		// Compute loss in FP32 to avoid FP16 underflow: with B*out=262144 elements,
		// dL/dy ~ 7.6e-6 which is below FP16 min-normal (6.1e-5) and would flush to zero.
		auto loss = (encoded.to(torch::kFloat32) - encoded_target.to(torch::kFloat32)).pow(2).mean();

		loss.backward();
		torch::Tensor gradients = encoding->parameters()[0].grad();

		ASSERT_EQ(gradients.size(0), encoding->n_params());
		ASSERT_TRUE(gradients.abs().sum().item<float>() > 0.f);

		auto named_grads = encoding->named_parameters();

		auto named_modules = encoding->named_modules(std::string("0"));
		ASSERT_EQ(named_modules.size(), 1);
	}

	TEST(LocationEncoding, Training) {
		cudaFree(0);
		auto encoding = std::make_shared<LocationEncodingBridge>(12, 64);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		::tcnn::GPUMatrixDynamic<float> locations(3, 1024);
		::tcnn::GPUMatrix<float> transformed_locations(64, 1024);

		float d_raw_data[1024*3];
		for (int i = 0; i < 1024; i++) {
			d_raw_data[(i * 3) + 0] = i / 1024.f;
			d_raw_data[(i * 3) + 1] = i / 1024.f;
			d_raw_data[(i * 3) + 2] = i / 1024.f;
		}

		CUDA_CHECK_THROW(cudaMemcpyAsync(
			locations.data(),
			d_raw_data,
			locations.n_bytes(),        // Größe in Bytes
			cudaMemcpyHostToDevice, 
			0
		));

		float t_raw_data[1024*64];
		for (int i = 0; i < 1024; i++) {
			for (int j = 0; j < 64; j++)
				t_raw_data[(i * 64) + j] = (i / 1024.f) * (j/64.f);
		}
		CUDA_CHECK_THROW(cudaMemcpyAsync(
			transformed_locations.data(),
			t_raw_data,
			transformed_locations.n_bytes(),        // Größe in Bytes
			cudaMemcpyHostToDevice, 
			0
		));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);

		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);

		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			encoding, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f;
		float last_loss = -1.f;
		for (int step = 0; step < 1000; step++) {
			auto ctx = trainer->training_step(0, locations, transformed_locations);

			if (step % 100 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (int l_idx = 0; l_idx < 1024; l_idx++)
					loss_val += cpu_losses[l_idx];
				loss_val /= 1024.f;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;
				last_loss = loss_val;
				if (first_loss < 0.f)
					first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss); 
	}

	TEST(ParameterSetEncoding, BridgeBackward) {
		cudaFree(0);
	
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		std::vector<rfnn::tcnn::ParameterSetEncoding::ParameterSet> sets = {
			{
				rfnn::tcnn::ParameterSetEncoding::ParameterSet(
					"spherical_harmonics",
					3
				)/*,
				rfnn::tcnn::ParameterSetEncoding::ParameterSet(
					"one_blob",
					1,
					16
				)*/
			}
		};

		auto encoding = std::make_shared<ParameterSetEncodingBridge>(sets, 64);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor dirs_and_dists = torch::rand({ 64 * 64, 3}, opts);
		ASSERT_EQ(dirs_and_dists.size(0), 64 * 64);
		ASSERT_EQ(dirs_and_dists.size(1), encoding->input_width());
		ASSERT_TRUE(dirs_and_dists.is_cuda());

		auto encoded = encoding->forward(dirs_and_dists);
		ASSERT_EQ(encoded.size(0), 64 * 64);
		ASSERT_TRUE(encoded.grad_fn());

		torch::Tensor encoded_target = torch::rand_like(encoded);
		// Compute loss in FP32 to avoid FP16 underflow: with B*out=262144 elements,
		// dL/dy ~ 7.6e-6 which is below FP16 min-normal (6.1e-5) and would flush to zero.
		auto loss = (encoded.to(torch::kFloat32) - encoded_target.to(torch::kFloat32)).pow(2).mean();

		loss.backward();
		torch::Tensor gradients = encoding->parameters()[0].grad();

		ASSERT_EQ(gradients.size(0), encoding->n_params());
		ASSERT_TRUE(gradients.abs().sum().item<float>() > 0.f);

		auto named_grads = encoding->named_parameters();

		auto named_modules = encoding->named_modules(std::string("0"));
		ASSERT_EQ(named_modules.size(), 1);
	}

	TEST(ParameterSetEncoding, Training) {
		cudaFree(0);
	
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		std::vector<rfnn::tcnn::ParameterSetEncoding::ParameterSet> sets = {
			{
				rfnn::tcnn::ParameterSetEncoding::ParameterSet(
					"spherical_harmonics",
					3
				),
				rfnn::tcnn::ParameterSetEncoding::ParameterSet(
					"one_blob",
					1,
					16
				)
			}
		};

		auto encoding = std::make_shared<ParameterSetEncodingBridge>(sets, 64);
		ASSERT_EQ(encoding->input_width(), 4u);

		// tcnn matrices are column-major (m = features, n = batch). We populate
		// them directly from host buffers; using a torch tensor's data_ptr() is
		// fragile because chained `.t().contiguous()` would dangle past the
		// expression and pack memory in an unexpected layout.
		::tcnn::GPUMatrixDynamic<float> dirs_and_dists_tcnn(4, 1024);
		::tcnn::GPUMatrix<float> transformed_tcnn(64, 1024);

		std::vector<float> dirs_host(4 * 1024);
		for (int i = 0; i < 1024; ++i) {
			dirs_host[i * 3 + 0] = 0.3f;
			dirs_host[i * 3 + 1] = -0.7f;
			dirs_host[i * 3 + 2] = 0.5f;
			dirs_host[i * 3 + 3] = 0.5f;
		}
		CUDA_CHECK_THROW(cudaMemcpy(dirs_and_dists_tcnn.data(), dirs_host.data(), dirs_and_dists_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		std::vector<float> target_host(64 * 1024);
		for (int j = 0; j < 64; ++j) {
			float v = 0.1f + 0.005f * j;
			for (int i = 0; i < 1024; ++i)
				target_host[i * 64 + j] = v;
		}
		CUDA_CHECK_THROW(cudaMemcpy(transformed_tcnn.data(), target_host.data(), transformed_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);

		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);

		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			encoding, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f;
		float last_loss = -1.f;
		for (int step = 0; step < 1000; step++) {
			auto ctx = trainer->training_step(0, dirs_and_dists_tcnn, transformed_tcnn);

			if (step % 100 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (int l_idx = 0; l_idx < 1024; l_idx++)
					loss_val += std::abs(cpu_losses[l_idx]);
				loss_val /= 1024.f;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;

				last_loss = loss_val;
				if (first_loss < 0.f)
					first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss);
	}

	TEST(FiLM, Training) {
		cudaFree(0);

		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t F = 64;   // feature channels
		constexpr uint32_t C = 8;    // condition channels
		constexpr uint32_t B = 1024; // batch size

		auto film = std::make_shared<FiLMBridge>(F, C, std::string("ReLU"));
		ASSERT_EQ(film->input_width(), F + C);
		ASSERT_EQ(film->output_width(), F);

		// tcnn matrices are column-major (m = channels, n = batch). The input
		// rows [0, F) hold the feature, rows [F, F+C) hold the condition.
		::tcnn::GPUMatrixDynamic<float> input_tcnn(F + C, B);
		::tcnn::GPUMatrix<float> target_tcnn(F, B);

		std::vector<float> input_host((F + C) * B);
		std::vector<float> target_host(F * B);

		// Build a deterministic, learnable mapping:
		//   feature[i,b] = 0.5 + 0.5 * sin(i * 0.13 + b * 0.007)
		//   condition[k,b] = 0.3 * cos(k * 0.71 + b * 0.011)
		//   target[i,b] = ReLU( (1 + g_i(condition_b)) * feature[i,b] + b_i(condition_b) )
		// where g_i / b_i are simple linear projections of the condition. This
		// is exactly the family FiLM can represent, so loss must drop sharply.
		auto g_coef = [](uint32_t i, uint32_t k) {
			return 0.4f * std::sin(0.31f * (float)i + 0.17f * (float)k);
		};
		auto b_coef = [](uint32_t i, uint32_t k) {
			return 0.2f * std::cos(0.23f * (float)i - 0.41f * (float)k);
		};

		for (uint32_t b = 0; b < B; ++b) {
			float cond[C];
			for (uint32_t k = 0; k < C; ++k) {
				cond[k] = 0.3f * std::cos(0.71f * (float)k + 0.011f * (float)b);
				input_host[b * (F + C) + F + k] = cond[k];
			}
			for (uint32_t i = 0; i < F; ++i) {
				const float feat = 0.5f + 0.5f * std::sin(0.13f * (float)i + 0.007f * (float)b);
				input_host[b * (F + C) + i] = feat;

				float gamma_raw = 0.f;
				float beta_raw  = 0.f;
				for (uint32_t k = 0; k < C; ++k) {
					gamma_raw += g_coef(i, k) * cond[k];
					beta_raw  += b_coef(i, k) * cond[k];
				}
				float pre = (1.f + gamma_raw) * feat + beta_raw;
				target_host[b * F + i] = pre > 0.f ? pre : 0.f;
			}
		}

		CUDA_CHECK_THROW(cudaMemcpy(input_tcnn.data(),  input_host.data(),  input_tcnn.n_bytes(),  cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(target_tcnn.data(), target_host.data(), target_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);

		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);

		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			film, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f;
		float last_loss = -1.f;
		for (int step = 0; step < 1000; step++) {
			auto ctx = trainer->training_step(0, input_tcnn, target_tcnn);

			if (step % 100 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (uint32_t l_idx = 0; l_idx < B; l_idx++)
					loss_val += std::abs(cpu_losses[l_idx]);
				loss_val /= (float)B;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;

				last_loss = loss_val;
				if (first_loss < 0.f)
					first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss);
		ASSERT_LT(last_loss, 0.5f * first_loss);
	}

	TEST(LayerNorm, ForwardSmoke) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		auto ln = std::make_shared<LayerNormBridge>(32u, 1e-5f);
		torch::TensorOptions opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
		torch::Tensor x = torch::rand({256, 32}, opts);
		torch::Tensor y = ln->forward<torch::Tensor>(x);
		cudaDeviceSynchronize();
		const float s = y.to(torch::kFloat32).abs().sum().item<float>();
		std::cout << "LayerNorm forward output abs().sum() = " << s << std::endl;
		ASSERT_TRUE(std::isfinite(s));
	}

	TEST(LayerNorm, Training) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t C = 32;
		constexpr uint32_t B = 1024;

		auto ln = std::make_shared<LayerNormBridge>(C, 1e-5f);

		::tcnn::GPUMatrixDynamic<float> in_tcnn(C, B);
		::tcnn::GPUMatrix<float> target_tcnn(C, B);

		std::vector<float> in_host(C * B);
		std::vector<float> target_host(C * B);

		// Inputs with random per-sample mean/scale; targets fix the desired
		// affine. After LayerNorm normalises the input, the learned gamma/beta
		// must reproduce the target_affine — which is the family LayerNorm can
		// exactly represent, so the loss must drop sharply.
		auto target_gamma = [](uint32_t i) { return 0.3f * std::sin(0.41f * (float)i); };
		auto target_beta  = [](uint32_t i) { return 0.2f * std::cos(0.27f * (float)i); };

		for (uint32_t b = 0; b < B; ++b) {
			const float mean  = 0.5f + 0.1f * std::sin(0.013f * (float)b);
			const float scale = 1.0f + 0.2f * std::cos(0.007f * (float)b);
			float sum = 0.f;
			float sq  = 0.f;
			for (uint32_t i = 0; i < C; ++i) {
				const float v = mean + scale * std::sin(0.31f * (float)i + 0.07f * (float)b);
				in_host[b * C + i] = v;
				sum += v;
				sq  += v * v;
			}
			const float mu  = sum / (float)C;
			const float var = sq / (float)C - mu * mu;
			const float inv_std = 1.f / std::sqrt(var + 1e-5f);
			for (uint32_t i = 0; i < C; ++i) {
				const float xn = (in_host[b * C + i] - mu) * inv_std;
				target_host[b * C + i] = xn * (1.f + target_gamma(i)) + target_beta(i);
			}
		}

		CUDA_CHECK_THROW(cudaMemcpy(in_tcnn.data(),     in_host.data(),     in_tcnn.n_bytes(),     cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(target_tcnn.data(), target_host.data(), target_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);
		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);
		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			ln, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f, last_loss = -1.f;
		for (int step = 0; step < 500; step++) {
			auto ctx = trainer->training_step(0, in_tcnn, target_tcnn);
			if (step % 50 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (uint32_t l = 0; l < B; l++) loss_val += std::abs(cpu_losses[l]);
				loss_val /= (float)B;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;
				last_loss = loss_val;
				if (first_loss < 0.f) first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss);
		ASSERT_LT(last_loss, 0.5f * first_loss);
	}

	TEST(MODEL, ForwardSmoke) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t D_MODEL = 64;
		constexpr uint32_t SPECTRUM_DIM = 32;
		constexpr uint32_t B = 256;

		auto beam_enc = std::make_shared<PBRFBeamEncoderBridge>(SPECTRUM_DIM, D_MODEL);
		auto base_model     = std::make_shared<ModelMainBridge>(D_MODEL);

		ASSERT_EQ(beam_enc->input_width(),  3u + 1u + SPECTRUM_DIM);
		ASSERT_EQ(beam_enc->output_width(), D_MODEL);
		ASSERT_EQ(base_model->input_width(),      3u + D_MODEL);
		ASSERT_EQ(base_model->output_width(),     33u);

		// Encode a fixed beam config first.
		torch::TensorOptions f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
		torch::Tensor beam_input = torch::rand({(long)B, 3 + 1 + (long)SPECTRUM_DIM}, f32);
		torch::Tensor beam_encoded = beam_enc->forward<torch::Tensor>(beam_input);

		ASSERT_EQ(beam_encoded.size(0), (long)B);
		ASSERT_EQ(beam_encoded.size(1), (long)D_MODEL);
		const float beam_sum = beam_encoded.to(torch::kFloat32).abs().sum().item<float>();
		std::cout << "beam_encoded.abs().sum() = " << beam_sum << std::endl;
		ASSERT_TRUE(std::isfinite(beam_sum));

		// Feed [xyz, beam_encoded(as float)] through the main forward.
		torch::Tensor xyz   = torch::rand({(long)B, 3}, f32);
		torch::Tensor beam_f = beam_encoded.to(torch::kFloat32);
		torch::Tensor concat = torch::cat({xyz, beam_f}, /*dim=*/1).contiguous();
		torch::Tensor out = base_model->forward<torch::Tensor>(concat);

		ASSERT_EQ(out.size(0), (long)B);
		ASSERT_EQ(out.size(1), 33);
		const float out_sum = out.to(torch::kFloat32).abs().sum().item<float>();
		std::cout << "base_model forward out.abs().sum() = " << out_sum << std::endl;
		ASSERT_TRUE(std::isfinite(out_sum));
	}

	// Backward through the main JIT-fused BaseRadiationPredictionModel: feeds [xyz, beam_slice] in,
	// computes an L2 loss in FP32 (to avoid FP16 underflow at large batch), and
	// verifies that the parameter gradient has non-zero entries.
	TEST(MODEL, MainBackward) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t D_MODEL = 64;
		constexpr uint32_t B = 256;

		auto base_model = std::make_shared<ModelMainBridge>(D_MODEL);
		ASSERT_EQ(base_model->input_width(),  3u + D_MODEL);
		ASSERT_EQ(base_model->output_width(), 33u);

		torch::TensorOptions f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor x = torch::rand({(long)B, 3 + (long)D_MODEL}, f32);

		auto y = base_model->forward<torch::Tensor>(x);
		ASSERT_EQ(y.size(0), (long)B);
		ASSERT_EQ(y.size(1), 33);
		ASSERT_TRUE(y.grad_fn());

		torch::Tensor target = torch::rand_like(y);
		auto loss = (y.to(torch::kFloat32) - target.to(torch::kFloat32)).pow(2).mean();
		loss.backward();

		torch::Tensor grads = base_model->parameters()[0].grad();
		ASSERT_EQ(grads.size(0), base_model->n_params());
		const float gsum = grads.to(torch::kFloat32).abs().sum().item<float>();
		std::cout << "MODEL main grads.abs().sum() = " << gsum << std::endl;
		ASSERT_TRUE(std::isfinite(gsum));
		ASSERT_GT(gsum, 0.f);
	}

	// Backward through the PBRFBeamEncoder (forward_impl/backward_impl, not
	// JIT-fused across the beam pipeline). Same flow as MainBackward.
	TEST(MODEL, BeamBackward) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t D_MODEL     = 64;
		constexpr uint32_t SPECTRUM_DIM = 32;
		constexpr uint32_t B = 256;

		auto beam = std::make_shared<PBRFBeamEncoderBridge>(SPECTRUM_DIM, D_MODEL);
		ASSERT_EQ(beam->input_width(),  3u + 1u + SPECTRUM_DIM);
		ASSERT_EQ(beam->output_width(), D_MODEL);

		torch::TensorOptions f32 = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32).requires_grad(true);
		torch::Tensor x = torch::rand({(long)B, 3 + 1 + (long)SPECTRUM_DIM}, f32);

		auto y = beam->forward<torch::Tensor>(x);
		ASSERT_EQ(y.size(0), (long)B);
		ASSERT_EQ(y.size(1), (long)D_MODEL);
		ASSERT_TRUE(y.grad_fn());

		torch::Tensor target = torch::rand_like(y);
		auto loss = (y.to(torch::kFloat32) - target.to(torch::kFloat32)).pow(2).mean();
		loss.backward();

		torch::Tensor grads = beam->parameters()[0].grad();
		ASSERT_EQ(grads.size(0), beam->n_params());
		const float gsum = grads.to(torch::kFloat32).abs().sum().item<float>();
		std::cout << "MODEL beam grads.abs().sum() = " << gsum << std::endl;
		ASSERT_TRUE(std::isfinite(gsum));
		ASSERT_GT(gsum, 0.f);
	}

	// End-to-end training of the JIT-fused PBRFBeamEncoder: the trainer
	// passes a single param buffer to set_params(), which must cascade to
	// LN1, MLP1, LN2, MLP2, ParameterSetEncoding. Every sub-module's
	// gradients are computed inside the fused backward kernel; a descending
	// loss proves the cascade is wired and the JIT chain trains all of them.
	TEST(MODEL, BeamTraining) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t SPECTRUM_DIM = 32;
		constexpr uint32_t D_MODEL      = 64;
		constexpr uint32_t B            = 1024;

		auto beam = std::make_shared<PBRFBeamEncoderBridge>(SPECTRUM_DIM, D_MODEL);

		::tcnn::GPUMatrixDynamic<float> in_tcnn(3 + 1 + SPECTRUM_DIM, B);
		::tcnn::GPUMatrix<float>        target_tcnn(D_MODEL, B);

		std::vector<float> in_host((3 + 1 + SPECTRUM_DIM) * B);
		std::vector<float> target_host(D_MODEL * B);
		for (uint32_t b = 0; b < B; ++b) {
			in_host[b * (3 + 1 + SPECTRUM_DIM) + 0] = std::sin(0.013f * (float)b);
			in_host[b * (3 + 1 + SPECTRUM_DIM) + 1] = std::cos(0.011f * (float)b);
			in_host[b * (3 + 1 + SPECTRUM_DIM) + 2] = std::sin(0.017f * (float)b);
			in_host[b * (3 + 1 + SPECTRUM_DIM) + 3] = 0.5f + 0.1f * std::sin(0.007f * (float)b);
			for (uint32_t i = 0; i < SPECTRUM_DIM; ++i)
				in_host[b * (3 + 1 + SPECTRUM_DIM) + 4 + i] = 0.3f * std::cos(0.71f * (float)i + 0.011f * (float)b);
			for (uint32_t j = 0; j < D_MODEL; ++j)
				target_host[b * D_MODEL + j] = 0.1f + 0.005f * (float)j;
		}
		CUDA_CHECK_THROW(cudaMemcpy(in_tcnn.data(),     in_host.data(),     in_tcnn.n_bytes(),     cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(target_tcnn.data(), target_host.data(), target_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);
		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);
		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			beam, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f, last_loss = -1.f;
		for (int step = 0; step < 300; step++) {
			auto ctx = trainer->training_step(0, in_tcnn, target_tcnn);
			if (step % 30 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (uint32_t l = 0; l < B; l++) loss_val += std::abs(cpu_losses[l]);
				loss_val /= (float)B;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;
				last_loss = loss_val;
				if (first_loss < 0.f) first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss);
	}

	// End-to-end training: drive BaseRadiationPredictionModel through the tcnn trainer to verify
	// the JIT-fused forward+backward path actually descends a loss surface.
	TEST(MODEL, MainTraining) {
		cudaFree(0);
		::tcnn::rtc_set_cache_dir("/mnt/data/repos/radfield3dnn-cpp/build/rtc");

		constexpr uint32_t D_MODEL = 64;
		constexpr uint32_t B = 1024;

		auto base_model = std::make_shared<ModelMainBridge>(D_MODEL);

		::tcnn::GPUMatrixDynamic<float> in_tcnn(3 + D_MODEL, B);
		::tcnn::GPUMatrix<float>        target_tcnn(33, B);

		std::vector<float> in_host((3 + D_MODEL) * B);
		std::vector<float> target_host(33 * B);
		for (uint32_t b = 0; b < B; ++b) {
			for (uint32_t i = 0; i < 3; ++i)
				in_host[b * (3 + D_MODEL) + i] = std::sin(0.13f * (float)i + 0.007f * (float)b);
			for (uint32_t k = 0; k < D_MODEL; ++k)
				in_host[b * (3 + D_MODEL) + 3 + k] = 0.3f * std::cos(0.71f * (float)k + 0.011f * (float)b);
			for (uint32_t i = 0; i < 33; ++i)
				target_host[b * 33 + i] = 0.1f + 0.005f * (float)i;
		}
		CUDA_CHECK_THROW(cudaMemcpy(in_tcnn.data(),     in_host.data(),     in_tcnn.n_bytes(),     cudaMemcpyHostToDevice));
		CUDA_CHECK_THROW(cudaMemcpy(target_tcnn.data(), target_host.data(), target_tcnn.n_bytes(), cudaMemcpyHostToDevice));

		auto optimizer = std::shared_ptr<::tcnn::Optimizer<::tcnn::network_precision_t>>(
			::tcnn::create_optimizer<::tcnn::network_precision_t>({{"otype", "Adam"}, {"learning_rate", 1e-3}})
		);
		auto loss = std::shared_ptr<::tcnn::Loss<::tcnn::network_precision_t>>(
			::tcnn::create_loss<::tcnn::network_precision_t>({{"otype", "L2"}})
		);
		auto trainer = std::make_shared<::tcnn::Trainer<float, ::tcnn::network_precision_t, ::tcnn::network_precision_t>>(
			base_model, optimizer, loss
		);

		cudaDeviceSynchronize();

		float first_loss = -1.f, last_loss = -1.f;
		for (int step = 0; step < 300; step++) {
			auto ctx = trainer->training_step(0, in_tcnn, target_tcnn);
			if (step % 30 == 0 || step == 0) {
				cudaDeviceSynchronize();
				auto cpu_losses = ctx->L.to_cpu_vector();
				float loss_val = 0.f;
				for (uint32_t l = 0; l < B; l++) loss_val += std::abs(cpu_losses[l]);
				loss_val /= (float)B;
				std::cout << "Step: " << step << " Loss: " << std::scientific << loss_val << std::endl;
				last_loss = loss_val;
				if (first_loss < 0.f) first_loss = loss_val;
			}
		}
		std::cout << "Finished training!" << std::endl;
		ASSERT_LT(last_loss, first_loss);
	}
};