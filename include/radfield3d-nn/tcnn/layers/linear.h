#pragma once
#include <memory>
#include <vector>
#include <string>
#include <tiny-cuda-nn/common.h>
#include <cuda_fp16.h>

namespace tcnn {
    template<typename T>
    class Encoding;

    template<typename T, typename B>
    class Network;

    template<typename T>
    class GPUMatrixDynamic;
};

namespace rfnn::tcnn {
    template<typename T>
    class Linear {
    public:
        Linear(size_t in_dim, size_t out_dim, const std::string& non_linearity = "None");
        ~Linear();
        std::shared_ptr<::tcnn::GPUMatrixDynamic<T>> forward(const ::tcnn::GPUMatrixDynamic<T>* x);
        size_t n_params() const;
        void set_params(T* params, T* inference_params, T* gradients);

    protected:
        const size_t in_dim;
        const size_t out_dim;
        void* handle;
        T* d_W;
        T* d_b;
    };

    extern template class Linear<float>;
    extern template class Linear<__half>;
};