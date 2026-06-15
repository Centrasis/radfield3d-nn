#include "radfield3d-nn-bindings/utils.h"
#include "radfield3d-nn/tcnn/encodings/global_parameters.h"
#include "radfield3d-nn/tcnn/encodings/location_encoding.h"
#include "radfield3d-nn/tcnn/encodings/beam_encoder.h"
#include "radfield3d-nn/tcnn/encodings/sperf_beam_encoder.h"
#include "radfield3d-nn/tcnn/layers/film.h"
#include "radfield3d-nn/tcnn/layers/layer_norm.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include "radfield3d-nn/tcnn/base_model.h"
#include <type_traits>
#include <cuda_fp16.h>
#include <torch/custom_class.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


::tcnn::network_precision_t LOSS_SCALE = static_cast<::tcnn::network_precision_t>(128.f);


namespace rfnn::tcnn {
	namespace kernels {
		namespace conversions {
			__global__ void float2half_forward_kernel(
				const float* __restrict__ x,
				__half* __restrict__ out,
				uint32_t N
			) {
				uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
				if (i >= N) return;

				out[i] = __float2half(x[i]);
			};
		};
	};
};

TORCH_LIBRARY(rfnn, m) {
  m.class_<rfnn::tcnn::autograd::TcnnState>("TcnnState");
};


std::shared_ptr<::tcnn::GPUMatrixDynamic<__half>> tcnnconvert::from_tensor2d2half(torch::Tensor x)
{
	// torch tensors are (B, C) row-major. tcnn's JIT kernel reads inputs via
	// MatrixView::col<C>(b), which expects sample b's C features to live in
	// contiguous memory at offsets [b*C, b*C+C-1]. That layout is exactly a
	// (B, C) row-major tensor wrapped as GPUMatrix(M=C, N=B, CM, stride=C).
	int padded = ::tcnn::next_multiple<int>(x.size(1), 16);
	int padding = padded - x.size(1);
	if (padding > 0)
		x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({ 0, padded - padding }));
	x = x.contiguous();
	assert(x.device().is_cuda());
	assert(x.is_contiguous());
	uint32_t batch_size = static_cast<uint32_t>(x.size(0));
	uint32_t feature_dims = static_cast<uint32_t>(x.size(1));

	if (x.dtype() == torch::kFloat32) {
		// Create the output half tensor with padding included
		auto half_tensor = std::make_shared<::tcnn::GPUMatrixDynamic<__half>>(feature_dims, batch_size);
		auto flt_tensor = ::tcnn::GPUMatrixDynamic<float>(x.data_ptr<float>(), feature_dims, batch_size);

		constexpr uint32_t BLOCK = 256;
		const uint32_t N = batch_size * feature_dims;
		uint32_t grid = (N + BLOCK - 1) / BLOCK;
		rfnn::tcnn::kernels::conversions::float2half_forward_kernel<<<grid, BLOCK>>>(
			flt_tensor.data(),
			half_tensor->data(),
			N
		);
		return half_tensor;
	}
	else {
		if (x.dtype() == torch::kFloat16) {
			auto out = std::make_shared<::tcnn::GPUMatrixDynamic<__half>>((__half*)x.data_ptr<torch::Half>(), feature_dims, batch_size);
			return out;
		}
		else {
			throw std::runtime_error("Unsupported data type!");
		}
	}

	return std::shared_ptr<::tcnn::GPUMatrixDynamic<__half>>();
}

std::shared_ptr<::tcnn::GPUMatrixDynamic<float>> tcnnconvert::from_tensor2d(torch::Tensor x) {
	// See from_tensor2d2half for the layout argument: (B, C) row-major wraps
	// directly as GPUMatrix(M=C, N=B, CM, stride=C). No transpose required.
	x = x.contiguous();
	assert(x.device().is_cuda());
	assert(x.is_contiguous());
	// >= not >: in the fp16 build network_precision_t is __half (2 < 4) and in
	// the fp32 build it is float (4 == 4). The original strict '>' silently
	// made an fp32 (TCNN_HALF_PRECISION=OFF) build fail to compile here. This
	// path always feeds the network its input as float regardless of compute
	// precision, so equal sizes are valid.
	static_assert(sizeof(float) >= sizeof(::tcnn::network_precision_t));
	uint32_t batch_size = static_cast<uint32_t>(x.size(0));
	uint32_t feature_dims = static_cast<uint32_t>(x.size(1));
	auto out = std::make_shared<::tcnn::GPUMatrixDynamic<float>>(x.data_ptr<float>(), feature_dims, batch_size);
	return out;
}

template<typename T>
torch::Tensor tcnnconvert::to_tensor2d(::tcnn::GPUMatrixDynamic<T>* x, bool return_owning_copy, bool requires_grad) {
	static_assert(std::is_same<T, float>::value || std::is_same<T, __half>::value);
	constexpr auto dtype = (std::is_same<T, float>::value) ? torch::kFloat32 : torch::kFloat16;
	const auto opts = torch::TensorOptions().dtype(dtype).device(torch::kCUDA).requires_grad(requires_grad);

	auto t = torch::from_blob(
		x->data(),
		{ (int64_t)x->n(), (int64_t)x->m() },
		{ (int64_t)x->m(), (int64_t)1 },
		opts
	);
	if (return_owning_copy)
		t = t.clone();
	return t;
}

template torch::Tensor tcnnconvert::to_tensor2d<float>(::tcnn::GPUMatrixDynamic<float>*, bool, bool);
template torch::Tensor tcnnconvert::to_tensor2d<__half>(::tcnn::GPUMatrixDynamic<__half>*, bool, bool);


template<typename T, typename input_dtypeT>
template<class FwdArgT>
torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<T, input_dtypeT>::forward(torch::autograd::AutogradContext* ctx, rfnn::tcnn::autograd::TorchBridge* bridge, FwdArgT& input, torch::Tensor& weights) {
	//static_assert((sizeof...(_Args) == 1 || sizeof...(_Args) == 2) && (std::is_same_v<_Args, torch::Tensor> && ...), "GenericModelFunction::forward only accepts one or two torch::Tensors as arguments.");
	static_assert(std::is_same_v<FwdArgT, torch::Tensor> || std::is_base_of_v<rfnn::tcnn::torch_inference::BaseFwdArg, FwdArgT>, "GenericModelFunction::forward needs a torch::Tensor or derived object from rfnn::tcnn::torch_inference::BaseFwdArg as input.");
	static_assert(sizeof(::tcnn::network_precision_t) <= 4, "GenericModelFunction::forward only accepts FP16 or FP32 as input types.");
	static constexpr auto dtype = (sizeof(::tcnn::network_precision_t) == 2) ? torch::kFloat16 : torch::kFloat32;
	static constexpr auto input_dtype = (sizeof(input_dtypeT) == 2) ? torch::kFloat16 : torch::kFloat32;

	// store ctx as tensor, so pytorch can safely handle its memory
	auto ctx_options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
	ctx->set_materialize_grads(false);

	const at::cuda::CUDAGuard device_guard{weights.device()};
	cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
	auto y_options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
	torch::Tensor x = (std::is_same_v<FwdArgT, torch::Tensor>) ? *reinterpret_cast<torch::Tensor*>(&input) : reinterpret_cast<rfnn::tcnn::torch_inference::BaseFwdArg*>(&input)->fwd_tensor();
	x = x.contiguous();
	const uint32_t B = x.size(0);
	const uint32_t C = dynamic_cast<T*>(bridge)->padded_output_width();
	// Output is (B, C) row-major: sample b's C output features live at offsets
	// [b*C, b*C+C-1], matching tcnn's CM(M=C, N=B, stride=C) view used inside
	// the JIT kernel via set_col(b, output).
	torch::Tensor y = torch::empty({B, C}, y_options);


	::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t> y_tcnn = [&]() {
		if constexpr (dtype == torch::kFloat32) {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)y.data_ptr<float>(),
				C,
				B
			);
		} else {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)y.data_ptr<torch::Half>(),
				C,
				B
			);
		}
	}();

	auto x_tcnn = [&]() {
		if constexpr (input_dtype == torch::kFloat32) {
			return tcnnconvert::from_tensor2d(x);
		} else {
			return tcnnconvert::from_tensor2d2half(x);
		}
	}();

	std::unique_ptr<::tcnn::Context> tcnn_ctx = dynamic_cast<T*>(bridge)->forward(cuda_stream, *x_tcnn.get(), &y_tcnn, false, true);
	auto tcnn_state = c10::make_intrusive<rfnn::tcnn::autograd::TcnnState>(
		std::move(tcnn_ctx)
	);

	ctx->saved_data["ctx"] = tcnn_state;
	ctx->saved_data["bridge"] = reinterpret_cast<int64_t>(bridge);

	// force pytorch to acknowledge dependency
	y = y + weights.sum() * 0.0f;

	ctx->save_for_backward({x, weights, y});

	return y;
}

template<typename T, typename input_dtypeT>
torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<T, input_dtypeT>::backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
	static_assert(sizeof(::tcnn::network_precision_t) <= 4, "GenericModelFunction::forward only accepts FP16 or FP32 as input types.");
	static constexpr auto dtype = (sizeof(::tcnn::network_precision_t) == 2) ? torch::kFloat16 : torch::kFloat32;
	static constexpr auto input_dtype = (sizeof(input_dtypeT) == 2) ? torch::kFloat16 : torch::kFloat32;

	torch::Tensor loss_scale = torch::full({}, static_cast<float>(LOSS_SCALE), at::device(at::kCUDA).dtype(dtype)); 

	// Both dy and the saved tensors live in (B, C) row-major already (see
	// forward()). No transpose: wrap directly as GPUMatrix(M=C, N=B, CM).
	auto dy = grad_outputs[0].contiguous();
	if constexpr (dtype == torch::kFloat16)
		dy *= loss_scale; //Do gradient scaling

	auto saved = ctx->get_saved_variables();
	::tcnn::Context* tcnn_ctx = ctx->saved_data["ctx"].toCustomClass<rfnn::tcnn::autograd::TcnnState>()->tcnn_ctx.get();
	rfnn::tcnn::autograd::TorchBridge* bridge = reinterpret_cast<rfnn::tcnn::autograd::TorchBridge*>(ctx->saved_data["bridge"].toInt());
	torch::Tensor x 		= saved[0];
	torch::Tensor& weights 	= saved[1];
	torch::Tensor& output 	= saved[2];

	const at::cuda::CUDAGuard device_guard{weights.device()};
	cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();
	
	torch::Tensor& grad = bridge->torch_grad();
	//grad.zero_();
	assert(grad.size(0) == dynamic_cast<T*>(bridge)->n_params());
	
	torch::Tensor pt_dLx = torch::empty_like(x);
	// All activation tensors are (B, C) row-major; tcnn wants GPUMatrix(M=C, N=B, CM).
	::tcnn::GPUMatrixDynamic<input_dtypeT> tcnn_x(x.data_ptr<input_dtypeT>(), x.size(1), x.size(0));

	::tcnn::GPUMatrixDynamic<input_dtypeT> tcnn_dLx = [&]() {
		if constexpr (input_dtype == torch::kFloat32) {
			return ::tcnn::GPUMatrixDynamic<input_dtypeT>(
				(input_dtypeT*)pt_dLx.data_ptr<float>(),
				pt_dLx.size(1),
				pt_dLx.size(0)
			);
		} else {
			return ::tcnn::GPUMatrixDynamic<input_dtypeT>(
				(input_dtypeT*)pt_dLx.data_ptr<torch::Half>(),
				pt_dLx.size(1),
				pt_dLx.size(0)
			);
		}
	}();

	::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t> tcnn_output = [&]() {
		if constexpr (dtype == torch::kFloat32) {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)output.data_ptr<float>(),
				output.size(1),
				output.size(0)
			);
		} else {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)output.data_ptr<torch::Half>(),
				output.size(1),
				output.size(0)
			);
		}
	}();

	::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t> tcnn_dL_dparams = [&]() {
		if constexpr (dtype == torch::kFloat32) {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)grad.data_ptr<float>(),
				grad.size(0),
				1
			);
		} else {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)grad.data_ptr<torch::Half>(),
				grad.size(0),
				1
			);
		}
	}();

 	::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t> tcnn_dL_dy = [&]() {
		if constexpr (dtype == torch::kFloat32) {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)dy.data_ptr<float>(),
				dy.size(1),
				dy.size(0)
			);
		} else {
			return ::tcnn::GPUMatrixDynamic<::tcnn::network_precision_t>(
				(::tcnn::network_precision_t*)dy.data_ptr<torch::Half>(),
				dy.size(1),
				dy.size(0)
			);
		}
	}();

	dynamic_cast<T*>(bridge)->backward(
		cuda_stream,
		*tcnn_ctx,
		tcnn_x,			// unused
		tcnn_output,	// unused
		tcnn_dL_dy,
		&tcnn_dLx
	);

	// Return a freshly-allocated copy of the parameter gradient. The bridge's
	// `internal_grad` buffer is shared across every forward/backward call on
	// the same module — when the user splits a batch into chunks
	// (FeedforwardPointwiseModel.forward2volume) and runs one .backward() over
	// the resulting loss, every chunk's backward writes into this same buffer.
	// PyTorch's AccumulateGrad aliases the first contribution it receives, so
	// later chunks overwriting `internal_grad` corrupt the gradient that
	// already landed in weights.grad. Cloning decouples each return.
	auto param_grad = (dtype == torch::kFloat16) ? grad.div_(loss_scale).clone() : grad.clone();
	return {
		torch::Tensor(),	// mockup gradient for TorchBridge* argument of forward
		(dtype == torch::kFloat16) ? pt_dLx.div_(loss_scale) : pt_dLx,
		param_grad
	};
}

template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LocationEncoding, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::ParameterSetEncoding, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::FiLM, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LayerNorm, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::PBRFBeamEncoder, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::SPERFBeamEncoder, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);
template torch::Tensor rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::BaseRadiationPredictionModel, float>::forward<torch::Tensor>(torch::autograd::AutogradContext*, rfnn::tcnn::autograd::TorchBridge*, torch::Tensor&, torch::Tensor&);

template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LocationEncoding, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::ParameterSetEncoding, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::FiLM, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::LayerNorm, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::PBRFBeamEncoder, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::SPERFBeamEncoder, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
template torch::autograd::tensor_list rfnn::tcnn::autograd::GenericModelFunction<rfnn::tcnn::BaseRadiationPredictionModel, float>::backward(torch::autograd::AutogradContext*, torch::autograd::tensor_list);
