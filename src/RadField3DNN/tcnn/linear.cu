#include "radfield3d-nn/tcnn/layers/linear.h"
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

using namespace rfnn::tcnn;

__global__ void add_bias_fp32(float* Y, const float* b, size_t output_dim, size_t batch) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * output_dim;
	if (idx >= total) return;

	int col = idx % output_dim;

	Y[idx] += b[col];
}

__global__ void add_bias_fp16(__half* Y, const __half* b, size_t output_dim, size_t batch) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int total = batch * output_dim;

	int vec_idx = idx * 2;
	if (vec_idx >= total) return;

	half2* Y2 = reinterpret_cast<half2*>(Y);

	int col = vec_idx % output_dim;
	if (col + 1 < output_dim) {
		int col2 = col >> 1;
		const half2* b2 = reinterpret_cast<const half2*>(b);
		Y2[idx] = __hadd2(Y2[idx], b2[col2]);
	}
	else {
		Y[vec_idx] = __hadd(Y[vec_idx], b[col]);
	}
}

template<typename T>
inline Linear<T>::Linear(size_t in_dim, size_t out_dim, const std::string& non_linearity)
	: in_dim(in_dim), out_dim(out_dim)
{
	static_assert(std::is_same<T, float>::value || std::is_same<T, __half>::value,
		"LinearLayer only supports float or __half");

	cublasCreate((cublasHandle_t*)&handle);
	cublasSetStream((cublasHandle_t)handle, 0);
}

template<typename T>
Linear<T>::~Linear()
{
	cublasDestroy((cublasHandle_t)handle);
}

template<typename T>
size_t Linear<T>::n_params() const
{
	return (in_dim * out_dim) + out_dim;
}

template<typename T>
void rfnn::tcnn::Linear<T>::set_params(T* params, T* inference_params, T* gradients)
{
	d_W = params;
	d_b = (params + (in_dim * out_dim));
}

template<typename T>
std::shared_ptr<::tcnn::GPUMatrixDynamic<T>> rfnn::tcnn::Linear<T>::forward(const ::tcnn::GPUMatrixDynamic<T>* x)
{
	const uint32_t B = x->n();
	const uint32_t C = x->m();
	assert(C == in_dim);
	assert((out_dim * B) % 2 == 0);	// Ensure power of 2 inputs

	std::shared_ptr<::tcnn::GPUMatrixDynamic<T>> Y = std::make_shared<::tcnn::GPUMatrixDynamic<T>>(out_dim, B);

	float alpha = 1.0, beta = 0.0;

	if constexpr (std::is_same<T, float>::value) {
		cublasSgemm(
			(cublasHandle_t)handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim, B, in_dim,
			&alpha,
			d_W, out_dim,
			x->data(), in_dim,
			&beta,
			Y->data(), out_dim
		);

		add_bias_fp32<<<(B * out_dim + 255) / 256, 256>>>(Y->data(), d_b, out_dim, B);
	}
	else {
		cublasGemmEx(
			(cublasHandle_t)handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			out_dim, B, in_dim,
			&alpha,
			d_W, CUDA_R_16F, out_dim,
			x->data(), CUDA_R_16F, in_dim,
			&beta,
			Y->data(), CUDA_R_16F, out_dim,
			CUBLAS_COMPUTE_32F,
			CUBLAS_GEMM_DEFAULT
		);

		add_bias_fp16<<<(static_cast<size_t>((B * out_dim) / 2) + 255) / 256, 256>>>(Y->data(), d_b, out_dim, B);
	}

	return Y;
}



template class Linear<float>;
template class Linear<__half>;