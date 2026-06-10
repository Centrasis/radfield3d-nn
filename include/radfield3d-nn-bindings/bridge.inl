#include <typeinfo>
#include <limits>
#include <torch/all.h>
#include <torch/types.h>
#include <torch/autograd.h>
#include <memory>
#include "radfield3d-nn-bindings/utils.h"


namespace rfnn::tcnn {
    namespace autograd {
        template<typename T, typename AutoGradFnT = rfnn::tcnn::autograd::GenericModelFunction<T>>
        class ModuleBridge : public T, public rfnn::tcnn::autograd::TorchBridge, public torch::nn::Module { // , public std::enable_shared_from_this<rfnn::tcnn::autograd::ModuleBridge<T, AutoGradFnT>> {
        protected:
            torch::Tensor weights;
            torch::Tensor internal_grad;
        
        public:
            template<class... _Args>
            explicit ModuleBridge(_Args&&... args) : T(std::forward<_Args>(args)...) {
                static constexpr auto dtype = (sizeof(::tcnn::network_precision_t) == 2) ? torch::kFloat16 : torch::kFloat32;

                int64_t num_parameters = static_cast<int64_t>(this->n_params());
                this->weights = register_parameter("weights", torch::empty(
                    { num_parameters },
                    torch::TensorOptions().dtype(dtype).device(torch::kCUDA).requires_grad(true)
                ));
                {
                    torch::NoGradGuard no_grad;
                    // MUST be zero-filled, not empty: tcnn's per-module
                    // initialize_params only writes each submodule's trainable
                    // weight matrices. Biases, non-self-initializing encoding
                    // tables and the inter-module padding slots are left as-is
                    // and are expected to be 0 (tcnn's Trainer memsets the
                    // param buffer to 0 before init, see trainer.h). With
                    // torch::empty those slots held garbage -> outputs
                    // exploded to ~1e3 and overflowed fp16 to NaN.
                    auto params_fp32 = torch::zeros(
                        { num_parameters },
                        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA)
                    );
                    uint64_t rng_seed = static_cast<uint64_t>(
                        torch::randint(0, std::numeric_limits<int64_t>::max(), { 1 },
                                       torch::TensorOptions().dtype(torch::kInt64)).item<int64_t>());
                    ::tcnn::pcg32 rnd{ rng_seed };
                    this->initialize_params(rnd, params_fp32.data_ptr<float>(), 1.0f);
                    // copy_ casts fp32 -> network_precision_t in place; the
                    // weights storage (and tcnn's raw pointer to it) is unchanged.
                    this->weights.copy_(params_fp32);
                }
                this->internal_grad = torch::zeros_like(this->weights);
                this->set_params(
                    [&]() { if constexpr (dtype == torch::kFloat32) { return (::tcnn::network_precision_t*)this->weights.data_ptr<float>(); } else { return (::tcnn::network_precision_t*)this->weights.data_ptr<torch::Half>(); } }(),
                    [&]() { if constexpr (dtype == torch::kFloat32) { return (::tcnn::network_precision_t*)this->weights.data_ptr<float>(); } else { return (::tcnn::network_precision_t*)this->weights.data_ptr<torch::Half>(); } }(),
                    [&]() { if constexpr (dtype == torch::kFloat32) { return (::tcnn::network_precision_t*)this->internal_grad.data_ptr<float>(); } else { return (::tcnn::network_precision_t*)this->internal_grad.data_ptr<torch::Half>(); } }()
                );
            }
            ~ModuleBridge() {};

            std::string name() const {
                return T::name(); 
            }

            virtual torch::Tensor& torch_weights() override {
                return this->weights;
            }

            virtual torch::Tensor& torch_grad() override {
                return this->internal_grad;
            }

            const std::vector<at::Tensor> parameters(bool recurse = true) const {
                return torch::nn::Module::parameters(recurse);
            }

            virtual void update_grad() override {
                //this->weights.mutable_grad() = this->internal_grad.to(torch::kFloat32);
                if (!this->weights.grad().defined()) {
                    this->weights.mutable_grad() = this->internal_grad;
                } else {
                    this->weights.grad().add_(this->internal_grad);
                }
            }

            //template<class... _Args>
            template<class ArgT>
            torch::Tensor forward(ArgT& x) {
                return AutoGradFnT::apply(dynamic_cast<rfnn::tcnn::autograd::TorchBridge*>(this), x, this->weights);
            }

            virtual void accumulate_gradients(torch::Tensor grad) override {
                if (this->internal_grad.defined())
                    this->internal_grad.add_(grad);
                else
                    this->internal_grad.copy_(grad);
                
                    this->update_grad();
            };

            virtual void zero_grad(bool set_to_none = true) override {
                // Never destroy internal_grad: m_gradients points to its data and
                // resetting the tensor leaves tcnn with a dangling pointer on next backward.
                if (this->internal_grad.defined())
                    this->internal_grad.zero_();
            };
        };
    };
};