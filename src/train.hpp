// Training loop with early stopping for the Era Translator DANN.
#pragma once

#include <torch/torch.h>

#include <functional>
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

// Everything the training loop knows at the end of one epoch. The GUI plots
// these live; the CLI ignores them and reads History at the end as before.
struct EpochUpdate {
    int epoch = 0;
    double lambda = 0.0;
    double lr = 0.0;
    double train_loss = 0.0;
    double val_loss = 0.0;
    double val_era_acc = 0.0;
    bool improved = false;  // this epoch became the new best_state
};

// Return false to stop training after the current epoch (cancel). best_state is
// still restored, so a cancelled run leaves a usable model.
using EpochCallback = std::function<bool(const EpochUpdate&)>;

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
                        torch::Device device = torch::kCPU, EpochCallback on_epoch = nullptr);

}  // namespace era
