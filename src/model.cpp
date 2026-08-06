#include "model.hpp"

namespace era {

torch::Tensor GradientReversalFunction::forward(torch::autograd::AutogradContext* ctx,
                                                torch::Tensor x, double lambda_) {
    ctx->saved_data["lambda"] = lambda_;
    return x.clone();
}

torch::autograd::tensor_list GradientReversalFunction::backward(
    torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs) {
    const double lambda_ = ctx->saved_data["lambda"].toDouble();
    // Second entry corresponds to lambda_, which takes no gradient.
    return {-lambda_ * grad_outputs[0], torch::Tensor()};
}

EraTranslatorDANNImpl::EraTranslatorDANNImpl(int64_t input_dim, int64_t num_eras)
    : num_eras(num_eras), latent_dim(LATENT_DIM) {
    encoder = register_module(
        "encoder", torch::nn::Sequential(
                       torch::nn::Linear(input_dim, ENCODER_HIDDEN), torch::nn::ReLU(),
                       torch::nn::Dropout(DROPOUT),
                       torch::nn::Linear(ENCODER_HIDDEN, LATENT_DIM)
                       // No final ReLU: a linear latent keeps all dims alive (ReLU dead
                       // units collapsed the 16-dim latent to a ~1-dim value axis).
                       ));
    value_head = register_module(
        "value_head", torch::nn::Sequential(torch::nn::Linear(LATENT_DIM, VALUE_HEAD_HIDDEN),
                                            torch::nn::ReLU(),
                                            torch::nn::Linear(VALUE_HEAD_HIDDEN, 1)));
    grl = register_module("grl", GradientReversalLayer());
    era_head = register_module(
        "era_head", torch::nn::Sequential(torch::nn::Linear(LATENT_DIM, ERA_HEAD_HIDDEN),
                                          torch::nn::ReLU(),
                                          torch::nn::Linear(ERA_HEAD_HIDDEN, ERA_HEAD_HIDDEN),
                                          torch::nn::ReLU(),
                                          torch::nn::Linear(ERA_HEAD_HIDDEN, num_eras)));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> EraTranslatorDANNImpl::forward(
    torch::Tensor x, double lambda_) {
    torch::Tensor z = encoder->forward(x);
    torch::Tensor value = value_head->forward(z).squeeze(-1);
    grl->lambda_ = lambda_;
    torch::Tensor era_logits = era_head->forward(grl->forward(z));
    return {z, value, era_logits};
}

}  // namespace era
