#include "train.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>

#include "config.hpp"

namespace era {
namespace {

// Pull each era's latent centroid toward the global centroid (CAT-style,
// arXiv:2407.12782). Directly minimizes era separability in the latent,
// bypassing the discriminator-collapse failure mode of GRL training.
torch::Tensor centroid_align_loss(const torch::Tensor& z, const torch::Tensor& era_labels) {
    const torch::Tensor normalized =
        torch::nn::functional::normalize(z, torch::nn::functional::NormalizeFuncOptions().dim(1));
    const torch::Tensor global_centroid = normalized.mean(0);
    torch::Tensor total = torch::zeros({}, normalized.options());
    for (int64_t e = 0; e < NUM_ERAS; ++e) {
        const torch::Tensor mask = era_labels == e;
        if (mask.sum().item<int64_t>() < 2) {
            continue;
        }
        const torch::Tensor centroid = normalized.index({mask}).mean(0);
        total = total + (centroid - global_centroid).pow(2).sum();
    }
    return total / NUM_ERAS;
}

}  // namespace

double lambda_at(int epoch, int max_epochs) {
    const double p = static_cast<double>(epoch) / std::max(max_epochs - 1, 1);
    return LAMBDA_MAX * (2.0 / (1.0 + std::exp(-LAMBDA_GAMMA * p)) - 1.0);
}

StateDict snapshot_state(const torch::nn::Module& module) {
    StateDict state;
    for (const auto& item : module.named_parameters(/*recurse=*/true)) {
        state[item.key()] = item.value().detach().to(torch::kCPU).clone();
    }
    for (const auto& item : module.named_buffers(/*recurse=*/true)) {
        state[item.key()] = item.value().detach().to(torch::kCPU).clone();
    }
    return state;
}

void restore_state(torch::nn::Module& module, const StateDict& state) {
    torch::NoGradGuard no_grad;
    for (auto& item : module.named_parameters(/*recurse=*/true)) {
        const auto it = state.find(item.key());
        if (it != state.end()) {
            item.value().copy_(it->second.to(item.value().device()));
        }
    }
    for (auto& item : module.named_buffers(/*recurse=*/true)) {
        const auto it = state.find(item.key());
        if (it != state.end()) {
            item.value().copy_(it->second.to(item.value().device()));
        }
    }
}

TrainResult train_model(EraTranslatorDANN& model, const torch::Tensor& X_train,
                        const torch::Tensor& y_train, const torch::Tensor& era_train,
                        const torch::Tensor& X_val, const torch::Tensor& y_val,
                        const torch::Tensor& era_val, int max_epochs, torch::Device device,
                        EpochCallback on_epoch) {
    torch::manual_seed(SEED);
    model->to(device);

    // Encoder + value head: SGD (smooth task-loss minima stabilize adversarial
    // training, Rangwani ICML 2022). Era head: its own Adam at higher LR so the
    // discriminator keeps pace with the encoder - a collapsed discriminator
    // (observed predicting a single decade) produces a degenerate reversal
    // gradient that never strips era from the latent.
    std::vector<torch::Tensor> task_params;
    for (const auto& p : model->encoder->parameters()) {
        task_params.push_back(p);
    }
    for (const auto& p : model->value_head->parameters()) {
        task_params.push_back(p);
    }
    torch::optim::SGD opt(
        task_params,
        torch::optim::SGDOptions(LR).momentum(MOMENTUM).weight_decay(WEIGHT_DECAY));
    torch::optim::Adam era_opt(model->era_head->parameters(),
                               torch::optim::AdamOptions(ERA_HEAD_LR));

    const torch::Tensor Xt = X_train.to(device);
    const torch::Tensor yt = y_train.to(device);
    const torch::Tensor et = era_train.to(device);
    const torch::Tensor Xv = X_val.to(device);
    const torch::Tensor yv = y_val.to(device);
    const torch::Tensor ev = era_val.to(device);

    // Pure inverse-frequency weighting: every era is equally present per batch,
    // so the era head cannot collapse to majority decades. LibTorch's C++ data
    // API has no WeightedRandomSampler, so the draw is done directly.
    const int64_t n_train = X_train.size(0);
    const torch::Tensor era_train_cpu = era_train.to(torch::kCPU).contiguous();
    const int64_t* era_train_data = era_train_cpu.data_ptr<int64_t>();
    std::vector<double> counts(NUM_ERAS, 0.0);
    for (int64_t i = 0; i < n_train; ++i) {
        counts[era_train_data[i]] += 1.0;
    }
    double inv_total = 0.0;
    for (const double c : counts) {
        inv_total += c > 0.0 ? 1.0 / c : 0.0;
    }
    std::vector<double> sample_weights(n_train);
    for (int64_t i = 0; i < n_train; ++i) {
        const double c = counts[era_train_data[i]];
        sample_weights[i] = c > 0.0 ? (1.0 / c) / inv_total : 0.0;
    }
    std::mt19937_64 rng(SEED);
    std::discrete_distribution<int64_t> sampler(sample_weights.begin(), sample_weights.end());

    torch::nn::MSELoss mse;
    torch::nn::CrossEntropyLoss ce;

    TrainResult result;
    double best_val_loss = std::numeric_limits<double>::infinity();
    int epochs_no_improve = 0;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        const double lam = lambda_at(epoch, max_epochs);
        // Ganin LR schedule: mu_p = LR / (1 + alpha*p)^beta
        const double p = static_cast<double>(epoch) / std::max(max_epochs - 1, 1);
        const double lr = LR / std::pow(1.0 + LR_ALPHA * p, LR_BETA);
        for (auto& group : opt.param_groups()) {
            static_cast<torch::optim::SGDOptions&>(group.options()).lr(lr);
        }

        std::vector<int64_t> draw(n_train);
        for (int64_t i = 0; i < n_train; ++i) {
            draw[i] = sampler(rng);
        }

        model->train();
        double total_loss = 0.0;
        for (int64_t start = 0; start < n_train; start += BATCH_SIZE) {
            const int64_t stop = std::min<int64_t>(start + BATCH_SIZE, n_train);
            const torch::Tensor batch_idx =
                torch::from_blob(draw.data() + start, {stop - start},
                                 torch::TensorOptions().dtype(torch::kInt64))
                    .clone()
                    .to(device);
            const torch::Tensor xb = Xt.index_select(0, batch_idx);
            const torch::Tensor yb = yt.index_select(0, batch_idx);
            const torch::Tensor eb = et.index_select(0, batch_idx);

            // Encoder/value step: MSE + GRL-era CE + CAT centroid alignment.
            opt.zero_grad();
            auto [z, vpred, epred] = model->forward(xb, lam);
            torch::Tensor loss = mse(vpred, yb) + lam * ce(epred, eb) +
                                 CENTROID_ALIGN_WEIGHT * centroid_align_loss(z, eb);
            loss.backward();
            opt.step();

            // Era head step (own optimizer; lambda_=0 disables reversal). Kept
            // purely as a monitoring head - its reversal gradient is unreliable
            // once it collapses, so era removal relies on the centroid loss.
            era_opt.zero_grad();
            auto [z2, vpred2, epred2] = model->forward(xb, 0.0);
            ce(epred2, eb).backward();
            era_opt.step();

            total_loss += loss.item<double>() * static_cast<double>(stop - start);
        }

        // Validation
        model->eval();
        double val_loss = 0.0;
        double val_era_acc = 0.0;
        {
            torch::NoGradGuard no_grad;
            auto [zv, vpred, epred] = model->forward(Xv, lam);
            val_loss = mse(vpred, yv).item<double>();
            val_era_acc = (epred.argmax(1) == ev).to(torch::kFloat).mean().item<double>();
        }

        const double train_loss = total_loss / static_cast<double>(n_train);
        result.history.train_loss.push_back(train_loss);
        result.history.val_loss.push_back(val_loss);
        result.history.val_era_acc.push_back(val_era_acc);

        bool improved = false;
        bool stop = false;
        if (val_loss < best_val_loss - 1e-5) {
            best_val_loss = val_loss;
            result.best_state = snapshot_state(*model);
            epochs_no_improve = 0;
            improved = true;
        } else if (epoch >= PATIENCE_WARMUP_EPOCHS) {
            // Early stopping only engages after the lambda schedule has ramped up.
            ++epochs_no_improve;
            if (epochs_no_improve >= PATIENCE) {
                stop = true;
            }
        }

        if (epoch % 10 == 0 || epoch == max_epochs - 1) {
            std::printf("epoch %3d | lam %.2f | train_loss %.4f | val_loss %.4f | val_era_acc %.4f\n",
                        epoch, lam, train_loss, val_loss, val_era_acc);
            std::fflush(stdout);
        }

        // Reported even on the early-stopping epoch, so an observer sees every
        // epoch that ran. A callback returning false cancels the run.
        if (on_epoch && !on_epoch(EpochUpdate{epoch, lam, lr, train_loss, val_loss, val_era_acc,
                                              improved})) {
            stop = true;
        }
        if (stop) {
            break;
        }
    }

    restore_state(*model, result.best_state);
    return result;
}

}  // namespace era
