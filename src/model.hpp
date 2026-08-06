// Domain-Adversarial Neural Network (DANN) for era-invariant embeddings.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <tuple>

#include "config.hpp"

namespace era {

// Identity in forward; gradient multiplied by -lambda_ in backward.
class GradientReversalFunction : public torch::autograd::Function<GradientReversalFunction> {
   public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x,
                                 double lambda_);
    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::tensor_list grad_outputs);
};

struct GradientReversalLayerImpl : torch::nn::Module {
    double lambda_ = 1.0;

    torch::Tensor forward(torch::Tensor x) { return GradientReversalFunction::apply(x, lambda_); }
};
TORCH_MODULE(GradientReversalLayer);

// Encoder -> [value head] and [GRL -> era head].
struct EraTranslatorDANNImpl : torch::nn::Module {
    explicit EraTranslatorDANNImpl(int64_t input_dim, int64_t num_eras = NUM_ERAS);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x,
                                                                    double lambda_ = 1.0);

    torch::nn::Sequential encoder{nullptr};
    torch::nn::Sequential value_head{nullptr};
    torch::nn::Sequential era_head{nullptr};
    GradientReversalLayer grl{nullptr};
    int64_t num_eras = NUM_ERAS;
    int64_t latent_dim = LATENT_DIM;
};
TORCH_MODULE(EraTranslatorDANN);

}  // namespace era
