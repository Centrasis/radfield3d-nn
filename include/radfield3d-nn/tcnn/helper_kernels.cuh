#pragma once
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/network.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>


namespace rfnn::tcnn {
    namespace kernels {
        /*namespace conversions {
            __global__ void float2half_forward_kernel(
                const float* __restrict__ x,
                __half* __restrict__ out,
                uint32_t N
            ) {
                uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= N) return;

                out[i] = __float2half(x[i]);
            };
        };*/

        namespace normalization {
            template <typename T>
            __global__ void layernorm_forward_kernel(
                const T* __restrict__ x,
                T* __restrict__ out,
                uint32_t C,
                uint32_t stride,
                float eps
            ) {
                uint32_t b = blockIdx.x; // one block per batch
                x += b * stride;
                out += b * stride;

                // Welford
                T mean = 0.0f;
                T m2 = 0.0f;
                uint32_t count = 0;

                for (uint32_t i = threadIdx.x; i < C; i += blockDim.x) {
                    T v = (T)x[i];
                    count++;
                    T delta = v - mean;
                    mean += delta / (T)count;
                    m2 += delta * (v - mean);
                }

                __shared__ T s_mean;
                __shared__ T s_var;

                // Reduce across threads
                using BlockReduce = cub::BlockReduce<T, 256>;
                using BlockReduceInt = cub::BlockReduce<uint32_t, 256>;
                __shared__ typename BlockReduce::TempStorage temp_storage;
                __shared__ typename BlockReduce::TempStorage temp_storageInt;


                mean = BlockReduce(temp_storage).Sum(mean);
                m2 = BlockReduce(temp_storage).Sum(m2);
                count = BlockReduce(temp_storageInt).Sum(count);

                if (threadIdx.x == 0) {
                    s_mean = mean / (T)count;
                    s_var = m2 / (T)count;
                }
                __syncthreads();

                T inv_std = (T)rsqrtf(s_var + (T)eps);

                // Normalize + affine
                for (uint32_t i = threadIdx.x; i < C; i += blockDim.x) {
                    T xn = ((T)x[i] - s_mean) * inv_std;
                    out[i] = (T)(xn);
                }
            }
        };
    };
};
