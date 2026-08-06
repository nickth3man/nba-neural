#include "evaluate.hpp"

#include <matplot/matplot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include "config.hpp"
#include "features.hpp"
#include "io.hpp"

namespace era {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The CSVs are written with '\n', but getline on a file saved elsewhere can
// still leave a trailing '\r' that would corrupt the last field.
void strip_cr(std::string& line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
}

// Rows centered on the global mean and scaled to unit length, so a dot product
// between two rows is the correlation similarity.
torch::Tensor centered_unit(const torch::Tensor& z) {
    const torch::Tensor centered = z - z.mean(0, /*keepdim=*/true);
    return centered / (centered.pow(2).sum(1, /*keepdim=*/true).sqrt() + 1e-12);
}

// ||yi - yj||^2 without materializing an (N, N, D) intermediate.
torch::Tensor pairwise_sq_dist(const torch::Tensor& Y) {
    const torch::Tensor sq = Y.pow(2).sum(1);
    return (sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * Y.mm(Y.t())).clamp_min(0.0);
}

double round_to(double value, int digits) {
    const double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

// Matplot++ stores colors as {alpha, r, g, b} in 0..1, where alpha 0 is opaque.
std::array<float, 4> hex_color(const std::string& hex) {
    const auto channel = [&hex](std::size_t offset) {
        return static_cast<float>(std::stoi(hex.substr(offset, 2), nullptr, 16)) / 255.0f;
    };
    return {0.0f, channel(1), channel(3), channel(5)};
}

torch::Tensor to_index_tensor(const std::vector<int64_t>& values) {
    return torch::from_blob(const_cast<int64_t*>(values.data()),
                            {static_cast<int64_t>(values.size())},
                            torch::TensorOptions().dtype(torch::kInt64))
        .clone();
}

std::vector<int64_t> embedding_era_codes(const Embeddings& emb) {
    std::vector<int64_t> codes(emb.era.size());
    for (std::size_t i = 0; i < emb.era.size(); ++i) {
        const auto it = std::find(ERA_ORDER.begin(), ERA_ORDER.end(), emb.era[i]);
        codes[i] = it == ERA_ORDER.end() ? -1 : std::distance(ERA_ORDER.begin(), it);
    }
    return codes;
}

}  // namespace

EvalResult evaluate_model(EraTranslatorDANN& model, const torch::Tensor& X, const torch::Tensor& y,
                          const torch::Tensor& era_labels, torch::Device device) {
    model->eval();
    torch::Tensor value_pred;
    torch::Tensor era_logits;
    {
        torch::NoGradGuard no_grad;
        auto [z, value, logits] = model->forward(X.to(device));
        value_pred = value.to(torch::kCPU).to(torch::kFloat64);
        era_logits = logits.to(torch::kCPU);
    }

    const torch::Tensor target = y.to(torch::kCPU).to(torch::kFloat64);
    EvalResult result;
    result.mse = (value_pred - target).pow(2).mean().item<double>();
    const double ss_res = (value_pred - target).pow(2).sum().item<double>();
    const double ss_tot = (target - target.mean()).pow(2).sum().item<double>();
    result.r2 = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 0.0;

    const torch::Tensor era_pred = era_logits.argmax(1);
    const torch::Tensor era_true = era_labels.to(torch::kCPU);
    result.era_acc = (era_pred == era_true).to(torch::kFloat64).mean().item<double>();

    std::vector<double> recalls;
    for (int64_t i = 0; i < static_cast<int64_t>(ERA_ORDER.size()); ++i) {
        const torch::Tensor mask = era_true == i;
        if (mask.sum().item<int64_t>() == 0) {
            continue;
        }
        const double recall =
            (era_pred.index({mask}) == i).to(torch::kFloat64).mean().item<double>();
        result.per_decade.emplace_back(ERA_ORDER[i], recall);
        recalls.push_back(recall);
    }
    // Balanced accuracy: macro-averaged per-decade recall. Collapse to a single
    // class yields 1/9 ~ 0.111, same as true random guessing.
    result.era_bal_acc =
        recalls.empty() ? 0.0
                        : std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
    return result;
}

Embeddings export_embeddings(const Frame& df, EraTranslatorDANN& model, const torch::Tensor& X,
                             const std::string& out_path, torch::Device device) {
    model->eval();
    torch::Tensor z;
    torch::Tensor value_pred;
    {
        torch::NoGradGuard no_grad;
        auto [latent, value, logits] = model->forward(X.to(device));
        z = latent.to(torch::kCPU).contiguous();
        value_pred = value.to(torch::kCPU).to(torch::kFloat64).contiguous();
    }

    Embeddings emb;
    emb.player_id = df.player_id;
    emb.player_name = df.player_name;
    emb.season_year = df.season_year;
    emb.era = df.era;
    emb.value_target = df.num.at("value_target");
    emb.value_pred.assign(value_pred.data_ptr<double>(),
                          value_pred.data_ptr<double>() + value_pred.numel());
    emb.z = z;

    std::vector<std::string> header = {"player_id", "player_name", "season_year", "era",
                                       "value_target", "value_pred"};
    for (int64_t d = 0; d < z.size(1); ++d) {
        header.push_back("z_" + std::to_string(d));
    }
    CsvWriter writer(out_path, header);
    const auto z_accessor = z.accessor<float, 2>();
    for (int64_t i = 0; i < emb.size(); ++i) {
        writer.write(emb.player_id[i]);
        writer.write(emb.player_name[i]);
        writer.write(emb.season_year[i]);
        writer.write(emb.era[i]);
        writer.write(emb.value_target[i]);
        // Prediction and latent are float32 in the model, so they are written at
        // float precision rather than widened to double.
        writer.write(static_cast<float>(emb.value_pred[i]));
        for (int64_t d = 0; d < z.size(1); ++d) {
            writer.write(z_accessor[i][d]);
        }
        writer.end_row();
    }
    return emb;
}

Embeddings read_embeddings_csv(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot read " + path);
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("empty embeddings file: " + path);
    }
    strip_cr(line);
    const std::vector<std::string> header = csv_split_line(line);
    // Six identity/value columns, then one column per latent dimension.
    constexpr std::size_t kIdentityCols = 6;
    if (header.size() <= kIdentityCols || header[0] != "player_id" || header[5] != "value_pred") {
        throw std::runtime_error("not an embeddings CSV: " + path);
    }
    const int64_t latent_dim = static_cast<int64_t>(header.size() - kIdentityCols);

    Embeddings emb;
    std::vector<float> latent;
    int64_t rows = 0;
    while (std::getline(in, line)) {
        strip_cr(line);
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> f = csv_split_line(line);
        if (f.size() != header.size()) {
            throw std::runtime_error(path + ": row " + std::to_string(rows + 1) + " has " +
                                     std::to_string(f.size()) + " fields, expected " +
                                     std::to_string(header.size()));
        }
        emb.player_id.push_back(std::stoll(f[0]));
        emb.player_name.push_back(f[1]);
        emb.season_year.push_back(f[2]);
        emb.era.push_back(f[3]);
        emb.value_target.push_back(std::stod(f[4]));
        emb.value_pred.push_back(std::stod(f[5]));
        for (std::size_t d = kIdentityCols; d < f.size(); ++d) {
            latent.push_back(std::stof(f[d]));
        }
        ++rows;
    }

    emb.z = torch::from_blob(latent.data(), {rows, latent_dim},
                             torch::TensorOptions().dtype(torch::kFloat32))
                .clone();
    return emb;
}

torch::Tensor cosine_sim_matrix(const torch::Tensor& z) {
    const torch::Tensor unit = centered_unit(z);
    return unit.mm(unit.t());
}

std::vector<ComparisonRow> write_comparisons(const Embeddings& emb, const std::string& out_path,
                                             int top_k) {
    const torch::Tensor unit = centered_unit(emb.z).to(torch::kFloat32).contiguous();
    const std::vector<int64_t> era_code = embedding_era_codes(emb);
    const torch::Tensor player_id = to_index_tensor(emb.player_id);

    // Index tensors per decade, so a query only scores the decade it is asking about.
    std::vector<torch::Tensor> era_rows(ERA_ORDER.size());
    std::vector<torch::Tensor> era_players(ERA_ORDER.size());
    for (std::size_t e = 0; e < ERA_ORDER.size(); ++e) {
        std::vector<int64_t> rows;
        for (int64_t i = 0; i < emb.size(); ++i) {
            if (era_code[i] == static_cast<int64_t>(e)) {
                rows.push_back(i);
            }
        }
        era_rows[e] = to_index_tensor(rows);
        era_players[e] = player_id.index_select(0, era_rows[e]);
    }

    const auto modern_it = std::find(ERA_ORDER.begin(), ERA_ORDER.end(), "2020s");
    const int64_t modern = std::distance(ERA_ORDER.begin(), modern_it);
    std::vector<int64_t> queries;
    for (int64_t i = 0; i < emb.size(); ++i) {
        if (era_code[i] == modern) {
            queries.push_back(i);
        }
    }

    std::vector<ComparisonRow> out;
    CsvWriter writer(out_path, {"query_player", "query_season", "query_value", "match_era",
                                "match_player", "match_season", "match_value", "cosine_sim"});

    // The Python built the full N x N similarity matrix, which is ~4.4 GB at
    // 23,516 rows. Scoring one chunk of queries against everything is equivalent
    // and bounded.
    constexpr int64_t kChunk = 256;
    for (std::size_t start = 0; start < queries.size(); start += kChunk) {
        const std::size_t stop = std::min<std::size_t>(start + kChunk, queries.size());
        const std::vector<int64_t> chunk(queries.begin() + start, queries.begin() + stop);
        const torch::Tensor sims = unit.index_select(0, to_index_tensor(chunk)).mm(unit.t());

        for (std::size_t c = 0; c < chunk.size(); ++c) {
            const int64_t i = chunk[c];
            const torch::Tensor sim_row = sims[static_cast<int64_t>(c)];
            for (std::size_t e = 0; e < ERA_ORDER.size(); ++e) {
                if (static_cast<int64_t>(e) == modern || era_rows[e].numel() == 0) {
                    continue;
                }
                // Exclude the query player's own other seasons.
                const torch::Tensor keep = era_players[e] != emb.player_id[i];
                torch::Tensor scores = sim_row.index_select(0, era_rows[e]);
                scores = torch::where(keep, scores, torch::full_like(scores, -kInf));

                const int64_t k = std::min<int64_t>(top_k, scores.numel());
                auto [values, local] = scores.topk(k);
                for (int64_t r = 0; r < k; ++r) {
                    const double score = values[r].item<double>();
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    const int64_t j = era_rows[e][local[r].item<int64_t>()].item<int64_t>();
                    ComparisonRow row;
                    row.query_player = emb.player_name[i];
                    row.query_season = emb.season_year[i];
                    row.query_value = round_to(emb.value_target[i], 3);
                    row.match_era = ERA_ORDER[e];
                    row.match_player = emb.player_name[j];
                    row.match_season = emb.season_year[j];
                    row.match_value = round_to(emb.value_target[j], 3);
                    row.cosine_sim = round_to(score, 4);

                    writer.write(row.query_player);
                    writer.write(row.query_season);
                    writer.write(row.query_value);
                    writer.write(row.match_era);
                    writer.write(row.match_player);
                    writer.write(row.match_season);
                    writer.write(row.match_value);
                    writer.write(row.cosine_sim);
                    writer.end_row();
                    out.push_back(std::move(row));
                }
            }
        }
    }
    return out;
}

double era_probe_balanced(const Embeddings& emb, int64_t seed) {
    const torch::Tensor Z = emb.z.to(torch::kFloat64);
    const std::vector<int64_t> codes = embedding_era_codes(emb);
    const torch::Tensor labels = to_index_tensor(codes);

    std::vector<int64_t> all(emb.size());
    std::iota(all.begin(), all.end(), 0);
    std::mt19937_64 rng(static_cast<uint64_t>(seed));
    auto [train_idx, test_idx] = stratified_split(all, emb.era, 0.3, rng);

    const torch::Tensor Ztr = Z.index_select(0, to_index_tensor(train_idx));
    const torch::Tensor Zte = Z.index_select(0, to_index_tensor(test_idx));
    const torch::Tensor ytr = labels.index_select(0, to_index_tensor(train_idx));
    const torch::Tensor yte = labels.index_select(0, to_index_tensor(test_idx));

    // Multinomial logistic regression matching sklearn's objective:
    // 0.5 * ||W||^2 + C * sum(cross entropy), with C = 1 and no penalty on the bias.
    torch::nn::Linear classifier(torch::nn::LinearOptions(Z.size(1), NUM_ERAS));
    classifier->to(torch::kFloat64);
    torch::optim::LBFGS optimizer(classifier->parameters(),
                                  torch::optim::LBFGSOptions(1.0).max_iter(2000));
    torch::nn::CrossEntropyLoss ce(
        torch::nn::CrossEntropyLossOptions().reduction(torch::kSum));

    const auto closure = [&]() {
        optimizer.zero_grad();
        torch::Tensor loss =
            ce(classifier->forward(Ztr), ytr) + 0.5 * classifier->weight.pow(2).sum();
        loss.backward();
        return loss;
    };
    optimizer.step(closure);

    torch::Tensor predictions;
    {
        torch::NoGradGuard no_grad;
        predictions = classifier->forward(Zte).argmax(1);
    }

    // Balanced accuracy = macro-averaged recall.
    std::vector<double> recalls;
    for (int64_t e = 0; e < NUM_ERAS; ++e) {
        const torch::Tensor mask = yte == e;
        if (mask.sum().item<int64_t>() == 0) {
            continue;
        }
        recalls.push_back(
            (predictions.index({mask}) == e).to(torch::kFloat64).mean().item<double>());
    }
    return recalls.empty() ? 0.0
                           : std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();
}

std::vector<SimilarMatch> query_similar(const Embeddings& emb, const std::string& player_name,
                                        const std::string& season_year, int top_k) {
    int64_t query = -1;
    for (int64_t i = 0; i < emb.size(); ++i) {
        if (emb.player_name[i] == player_name && emb.season_year[i] == season_year) {
            query = i;
            break;
        }
    }
    if (query < 0) {
        throw std::runtime_error("No embedding for " + player_name + " " + season_year);
    }

    const torch::Tensor unit = centered_unit(emb.z);
    torch::Tensor scores = unit.mv(unit[query]).to(torch::kFloat64);
    const int64_t query_player = emb.player_id[query];
    const torch::Tensor keep = to_index_tensor(emb.player_id) != query_player;
    scores = torch::where(keep, scores, torch::full_like(scores, -kInf));

    const int64_t k = std::min<int64_t>(top_k, scores.numel());
    auto [values, order] = scores.topk(k);
    std::vector<SimilarMatch> out;
    for (int64_t r = 0; r < k; ++r) {
        const int64_t j = order[r].item<int64_t>();
        out.push_back(SimilarMatch{emb.player_name[j], emb.season_year[j], emb.era[j],
                                   emb.value_target[j], emb.value_pred[j],
                                   values[r].item<double>()});
    }
    return out;
}

namespace {

// Conditional neighbour probabilities matched to the target perplexity by a
// per-point binary search on beta = 1 / (2 sigma^2), then symmetrized.
torch::Tensor joint_probabilities(const torch::Tensor& X, double perplexity) {
    torch::NoGradGuard no_grad;
    const int64_t n = X.size(0);
    const torch::Tensor D = pairwise_sq_dist(X);
    const double log_perplexity = std::log(perplexity);

    torch::Tensor beta = torch::ones({n}, X.options());
    torch::Tensor beta_min = torch::full({n}, -kInf, X.options());
    torch::Tensor beta_max = torch::full({n}, kInf, X.options());

    const auto conditional = [&](const torch::Tensor& b) {
        torch::Tensor p = torch::exp(-D * b.unsqueeze(1));
        p.diagonal().fill_(0.0);
        return p;
    };

    for (int iteration = 0; iteration < 50; ++iteration) {
        const torch::Tensor p = conditional(beta);
        const torch::Tensor sum_p = p.sum(1).clamp_min(1e-12);
        const torch::Tensor entropy = torch::log(sum_p) + beta * (D * p).sum(1) / sum_p;

        // Entropy above target means the distribution is too flat -> raise beta.
        const torch::Tensor too_flat = entropy > log_perplexity;
        beta_min = torch::where(too_flat, beta, beta_min);
        beta_max = torch::where(too_flat, beta_max, beta);
        beta = torch::where(too_flat,
                            torch::where(torch::isinf(beta_max), beta * 2.0,
                                         (beta + beta_max) / 2.0),
                            torch::where(torch::isinf(beta_min), beta / 2.0,
                                         (beta + beta_min) / 2.0));
    }

    torch::Tensor p = conditional(beta);
    p = p / p.sum(1, /*keepdim=*/true).clamp_min(1e-12);
    return ((p + p.t()) / (2.0 * static_cast<double>(n))).clamp_min(1e-12);
}

}  // namespace

torch::Tensor tsne(const torch::Tensor& X, int64_t seed) {
    const int64_t n = X.size(0);
    const torch::Tensor input = X.to(torch::kFloat32);
    const torch::Tensor P = joint_probabilities(input, 30.0);
    const torch::Tensor log_P = P.log();

    torch::manual_seed(seed);
    // PCA initialization scaled to std 1e-4, as sklearn's init="pca" does.
    torch::Tensor Y;
    {
        torch::NoGradGuard no_grad;
        const torch::Tensor centered = input - input.mean(0, /*keepdim=*/true);
        auto [U, S, Vh] = torch::linalg_svd(centered, /*full_matrices=*/false);
        Y = U.narrow(1, 0, 2) * S.narrow(0, 0, 2);
        Y = Y / Y.select(1, 0).std().clamp_min(1e-12) * 1e-4;
    }
    Y = Y.detach().clone().set_requires_grad(true);

    torch::optim::SGD optimizer({Y}, torch::optim::SGDOptions(200.0).momentum(0.5));
    constexpr int kIterations = 1000;
    constexpr int kExaggerationIterations = 250;
    constexpr double kExaggeration = 12.0;

    for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (iteration == kExaggerationIterations) {
            static_cast<torch::optim::SGDOptions&>(optimizer.param_groups()[0].options())
                .momentum(0.8);
        }
        const bool exaggerate = iteration < kExaggerationIterations;
        const torch::Tensor P_eff = exaggerate ? P * kExaggeration : P;
        const torch::Tensor log_P_eff =
            exaggerate ? log_P + std::log(kExaggeration) : log_P;

        // Student-t affinities; the diagonal is a constant 1 and contributes no
        // gradient, so it is simply subtracted out of the normalizer.
        const torch::Tensor num = 1.0 / (1.0 + pairwise_sq_dist(Y));
        const torch::Tensor Q =
            (num / (num.sum() - static_cast<double>(n))).clamp_min(1e-12);
        const torch::Tensor loss = (P_eff * (log_P_eff - Q.log())).sum();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    return Y.detach();
}

void plot_tsne(const Embeddings& emb, const std::string& out_path, int64_t sample, int64_t seed) {
    std::vector<int64_t> rows(emb.size());
    std::iota(rows.begin(), rows.end(), 0);
    if (emb.size() > sample) {
        std::mt19937_64 rng(static_cast<uint64_t>(seed));
        std::shuffle(rows.begin(), rows.end(), rng);
        rows.resize(sample);
    }

    const torch::Tensor Y = tsne(emb.z.index_select(0, to_index_tensor(rows)), seed).contiguous();
    const auto xy = Y.accessor<float, 2>();

    const std::vector<std::string> era_colors = {"#8c564b", "#e377c2", "#7f7f7f",
                                                 "#bcbd22", "#17becf", "#1f77b4",
                                                 "#ff7f0e", "#2ca02c", "#d62728"};

    auto figure = matplot::figure(true);
    figure->size(1000, 800);
    auto axes = figure->current_axes();
    axes->hold(matplot::on);

    std::vector<std::string> plotted;
    for (std::size_t e = 0; e < ERA_ORDER.size(); ++e) {
        std::vector<double> xs;
        std::vector<double> ys;
        for (std::size_t r = 0; r < rows.size(); ++r) {
            if (emb.era[rows[r]] == ERA_ORDER[e]) {
                xs.push_back(xy[static_cast<int64_t>(r)][0]);
                ys.push_back(xy[static_cast<int64_t>(r)][1]);
            }
        }
        if (xs.empty()) {
            continue;
        }
        const std::array<float, 4> color = hex_color(era_colors[e]);
        auto points = axes->scatter(xs, ys);
        points->marker_size(4);
        points->marker_color(color);
        points->marker_face_color(color);
        plotted.push_back(ERA_ORDER[e]);
    }

    axes->title("t-SNE of Era-Translator latent space (colored by decade)");
    matplot::legend(axes, plotted);

    // Matplot++ embeds the path in a double-quoted gnuplot string, where a
    // backslash is an escape - a Windows "out\tsne_era.png" silently becomes a
    // tab. Forward slashes work on both platforms.
    const std::string gnuplot_path = std::filesystem::path(out_path).generic_string();
    figure->save(gnuplot_path);

    // gnuplot failures do not propagate back through Matplot++, so the only
    // reliable signal is whether the file appeared.
    if (!std::filesystem::exists(out_path)) {
        throw std::runtime_error("gnuplot did not produce " + out_path);
    }
}

}  // namespace era
