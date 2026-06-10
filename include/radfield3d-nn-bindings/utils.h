#pragma once
#include <torch/all.h>
#include <torch/types.h>
#include <torch/autograd.h>
#include <memory>
#include "radfield3d-nn/tcnn/base_model.h"


namespace tcnnconvert {
	std::shared_ptr<::tcnn::GPUMatrixDynamic<__half>> from_tensor2d2half(torch::Tensor x);
	std::shared_ptr<::tcnn::GPUMatrixDynamic<float>> from_tensor2d(torch::Tensor x);

	template<typename T = ::tcnn::network_precision_t>
	torch::Tensor to_tensor2d(::tcnn::GPUMatrixDynamic<T>* x, bool return_owning_copy = false, bool requires_grad = false);
};

extern template torch::Tensor tcnnconvert::to_tensor2d<float>(::tcnn::GPUMatrixDynamic<float>*, bool, bool);
extern template torch::Tensor tcnnconvert::to_tensor2d<__half>(::tcnn::GPUMatrixDynamic<__half>*, bool, bool);

namespace rfnn::tcnn {
	namespace torch_inference {
        class BaseFwdArg {
        public:
            virtual ~BaseFwdArg();
            virtual torch::Tensor fwd_tensor() const = 0;
        };

        class ConditionalInput : public BaseFwdArg {
        protected:
            torch::Tensor buffer;
        public:
            ConditionalInput(torch::Tensor&& location, torch::Tensor&& condition) {
                assert(location.size(0) == condition.size(0));
                assert(location.options().dtype() == condition.options().dtype());
                const uint32_t B = location.size(0);
                this->buffer = torch::empty(
                    {
                        B,
                        location.size(1) + condition.size(1)
                    },
                    location.options()
                );

                this->buffer.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}, location);
                this->buffer.index_put_({torch::indexing::Slice(), torch::indexing::Slice(3, torch::indexing::None)}, condition);
            }

            virtual torch::Tensor fwd_tensor() const override { return this->buffer; }            
        };

        struct ModelCapsule : public c10::intrusive_ptr_target {
            std::shared_ptr<rfnn::tcnn::BaseRadiationPredictionModel> model;
            ModelCapsule(std::shared_ptr<rfnn::tcnn::BaseRadiationPredictionModel> m) : model(std::move(m)) {}
        };
	};

	namespace autograd {
        struct TcnnState : torch::CustomClassHolder {
            std::unique_ptr<::tcnn::Context> tcnn_ctx;
            
            TcnnState(std::unique_ptr<::tcnn::Context> tcnn_ctx) 
                : tcnn_ctx(std::move(tcnn_ctx)) {}
        };

		class TorchBridge {
		public:
			virtual void accumulate_gradients(torch::Tensor grad) = 0;
            virtual void zero_grad(bool set_to_none = true) = 0;
            virtual torch::Tensor& torch_weights() = 0;
            virtual void update_grad() = 0;
            virtual torch::Tensor& torch_grad() = 0;
		};

        template<typename T, typename input_dtypeT = ::tcnn::network_precision_t>
        struct GenericModelFunction : public torch::autograd::Function<GenericModelFunction<T, input_dtypeT>> {
            template<class FwdArgT>
            static torch::Tensor forward(torch::autograd::AutogradContext* ctx, rfnn::tcnn::autograd::TorchBridge* bridge, FwdArgT& input, torch::Tensor& weights);
            static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs);
        };
	};
};

namespace rfnn::tcnn {
    namespace torch_inference {
        template<typename T, typename fn>
        class TorchBridge : public rfnn::tcnn::autograd::TorchBridge, public std::enable_shared_from_this<TorchBridge<T, fn>> {
        protected:
            std::unique_ptr<T> tcnn_model;

            torch::Tensor w;
            torch::Tensor internal_grad;

        public:
            virtual void accumulate_gradients(torch::Tensor grad) override {
                if (w.mutable_grad().defined())
                    w.mutable_grad().add_(grad);
                else
                    w.mutable_grad().copy_(grad);
            }

            virtual void zero_grad(bool set_to_none = true) override {
                if (set_to_none) {
                    if (this->w.mutable_grad().defined())
                        this->w.mutable_grad().reset();
                }
                else {
                    if (this->w.mutable_grad().defined())
                        this->w.mutable_grad().zero_();
                    else {
                        this->w.mutable_grad().copy_(torch::zeros_like(this->w));
                    }
                }
            }

            template<class... _Args>
            explicit TorchBridge(_Args&&... args) : tcnn_model(std::make_unique<T>(std::forward<_Args>(args)...)) {
                static constexpr auto dtype = (sizeof(::tcnn::network_precision_t) == 2) ? torch::kFloat16 : torch::kFloat32;

                w = torch::zeros(
                    { static_cast<int64_t>(tcnn_model->n_params()) },
                    torch::TensorOptions().dtype(dtype).requires_grad(true).device(torch::kCUDA)
                );
                internal_grad = torch::zeros_like(w);
                
                tcnn_model->set_params(
                    (::tcnn::network_precision_t*)w.data_ptr(),
                    (::tcnn::network_precision_t*)w.data_ptr(),
                    (::tcnn::network_precision_t*)internal_grad.data_ptr()
                );
            }

            torch::Tensor weights() const {
                return w;
            }

            const torch::Tensor gradients() const {
                return w.grad();
            }

            template<class... _Args>
            torch::Tensor forward(_Args&&... args) {
                return fn::apply(std::forward<_Args>(args)..., this->tcnn_model.get(), this);
            }
        };
    };
};
