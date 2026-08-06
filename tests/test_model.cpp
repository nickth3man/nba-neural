// Unit tests for the DANN model components.
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "model.hpp"

TEST(Model, GrlForwardIdentityBackwardReversal) {
    era::GradientReversalLayer layer;
    layer->lambda_ = 2.0;
    const torch::Tensor x = torch::randn({4, 8}, torch::requires_grad());

    const torch::Tensor out = layer->forward(x);
    EXPECT_TRUE(torch::allclose(out, x));  // forward identity

    out.pow(2).sum().backward();
    // d/dx (x^2) = 2x; reversed by -lambda -> -4x
    EXPECT_TRUE(torch::allclose(x.grad(), -4.0 * x.detach(), 1e-5, 1e-5));
}

TEST(Model, DannForwardShapes) {
    era::EraTranslatorDANN model(13);
    const torch::Tensor x = torch::randn({8, 13});

    auto [z, value, era_logits] = model->forward(x, 1.0);
    EXPECT_EQ(z.sizes(), (std::vector<int64_t>{8, model->latent_dim}));
    EXPECT_EQ(value.sizes(), (std::vector<int64_t>{8}));
    EXPECT_EQ(era_logits.sizes(), (std::vector<int64_t>{8, model->num_eras}));
}

TEST(Model, DannTrainsOneStep) {
    torch::manual_seed(0);
    era::EraTranslatorDANN model(13);
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(1e-2));
    torch::nn::MSELoss mse;
    torch::nn::CrossEntropyLoss ce;

    const torch::Tensor x = torch::randn({16, 13});
    const torch::Tensor y = torch::randn({16});
    const torch::Tensor e = torch::randint(0, 9, {16}, torch::kInt64);

    auto [z, value, era_logits] = model->forward(x, 0.5);
    const torch::Tensor loss = mse(value, y) + 0.5 * ce(era_logits, e);
    loss.backward();
    optimizer.step();

    EXPECT_GT(loss.item<double>(), 0.0);
}
