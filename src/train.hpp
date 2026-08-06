// Training loop with early stopping for the Era Translator DANN.
#pragma once

#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

#include "model.hpp"

namespace era {

using StateDict = std::map<std::string, torch::Tensor>;

struct History {
    std::vector<double> train_loss;
    std::vector<double> val_loss;
    std::vector<double> val_era_acc;
};

struct TrainResult {
    History history;
    StateDict best_state;
};

// Ganin et al. sigmoid schedule: lambda_p = 2/(1+exp(-gamma*p)) - 1, p in [0,1].
// Ramps smoothly 0 -> LAMBDA_MAX over the whole run (arXiv:1505.07818 sec. 5.2.2).
double lambda_at(int epoch, int max_epochs);

StateDict snapshot_state(const torch::nn::Module& module);
void restore_state(torch::nn::Module& module, const StateDict& state);

// Train the DANN. Follows Ganin et al. (arXiv:1505.07818): sigmoid lambda schedule;
// the era head trains on era-balanced batches (inverse-frequency sampling) so
// majority decades cannot dominate it. Balanced batches only - no class-balanced
// CE weights, since Rangwani et al. (ICML 2022) show weakening the adversarial
// loss is harmful. SGD + momentum per Rangwani (smooth task-loss minima stabilize
// adversarial training) with Ganin LR decay.
TrainResult train_model(EraTranslatorDANN& model, const torch::Tensor& X_train,
                        const torch::Tensor& y_train, const torch::Tensor& era_train,
                        const torch::Tensor& X_val, const torch::Tensor& y_val,
                        const torch::Tensor& era_val, int max_epochs = MAX_EPOCHS,
                        torch::Device device = torch::kCPU);

}  // namespace era
