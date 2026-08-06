// Evaluation: metrics, embedding export, t-SNE plot, cross-era comparisons.
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "frame.hpp"
#include "model.hpp"

namespace era {

struct EvalResult {
    double mse = 0.0;
    double r2 = 0.0;
    double era_acc = 0.0;
    double era_bal_acc = 0.0;
    std::vector<std::pair<std::string, double>> per_decade;
};

struct Embeddings {
    std::vector<int64_t> player_id;
    std::vector<std::string> player_name;
    std::vector<std::string> season_year;
    std::vector<std::string> era;
    std::vector<double> value_target;
    std::vector<double> value_pred;
    torch::Tensor z;  // (N, LATENT_DIM), float32, CPU

    int64_t size() const { return static_cast<int64_t>(player_id.size()); }
};

struct SimilarMatch {
    std::string player_name;
    std::string season_year;
    std::string era;
    double value_target = 0.0;
    double value_pred = 0.0;
    double cosine_sim = 0.0;
};

struct ComparisonRow {
    std::string query_player;
    std::string query_season;
    double query_value = 0.0;
    std::string match_era;
    std::string match_player;
    std::string match_season;
    double match_value = 0.0;
    double cosine_sim = 0.0;
};

// Return metrics on the given split: mse, r2, era accuracy, balanced era
// accuracy and per-decade recall.
EvalResult evaluate_model(EraTranslatorDANN& model, const torch::Tensor& X, const torch::Tensor& y,
                          const torch::Tensor& era_labels, torch::Device device = torch::kCPU);

// Write embeddings.csv with identity columns, target, prediction and z_0..z_23.
Embeddings export_embeddings(const Frame& df, EraTranslatorDANN& model, const torch::Tensor& X,
                             const std::string& out_path, torch::Device device = torch::kCPU);

// Correlation similarity: centered cosine (Pearson-like).
//
// Plain cosine collapses toward ~0.99 for all pairs once the centroid-alignment
// loss pulls the latent toward a shared direction (observed mean 0.987), which
// destroys player discrimination. Centering removes the shared direction; the
// resulting sims span [-1, 1] and recover meaningful cross-era matches.
torch::Tensor cosine_sim_matrix(const torch::Tensor& z);

// For each 2020s player-season, top-k most similar across every other decade.
// Excludes the query player's own other seasons (same player_id).
std::vector<ComparisonRow> write_comparisons(const Embeddings& emb, const std::string& out_path,
                                             int top_k = 5);

// Fresh multinomial logistic-regression era probe on frozen embeddings.
//
// The in-training era head collapses to predict-a-single-decade (balanced acc
// exactly 1/9) once the adversarial loss dominates, so it is NOT a valid
// invariance measure. A fresh classifier on held-out embeddings is the honest
// check: balanced accuracy near 1/9 means era info is genuinely stripped from z.
double era_probe_balanced(const Embeddings& emb, int64_t seed = 42);

// Top-k most similar player-seasons across all eras, excluding the query
// player's own other seasons (same player_id).
std::vector<SimilarMatch> query_similar(const Embeddings& emb, const std::string& player_name,
                                        const std::string& season_year, int top_k = 5);

// Exact t-SNE (perplexity 30) optimized with autograd on the KL objective.
torch::Tensor tsne(const torch::Tensor& X, int64_t seed = 42);

// 2-D t-SNE colored by era; decades should overlap if the space is invariant.
// Needs gnuplot on PATH (Matplot++ renders through it).
void plot_tsne(const Embeddings& emb, const std::string& out_path, int64_t sample = 4000,
               int64_t seed = 42);

}  // namespace era
