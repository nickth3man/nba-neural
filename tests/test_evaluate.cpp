// Unit tests for evaluation helpers.
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <string>

#include "config.hpp"
#include "evaluate.hpp"
#include "frame.hpp"
#include "model.hpp"

namespace {

era::Embeddings fake_embeddings(int n = 30) {
    torch::manual_seed(0);
    const std::vector<std::string> eras = {"1940s", "1990s", "2020s"};

    era::Embeddings emb;
    for (int i = 0; i < n; ++i) {
        emb.player_id.push_back(100 + (i % 6));
        emb.player_name.push_back("P" + std::to_string(i));
        emb.season_year.push_back(std::to_string(2000 + i) + "-01");
        emb.era.push_back(eras[i % 3]);
        emb.value_target.push_back(i);
        emb.value_pred.push_back(i);
    }
    emb.z = torch::randn({n, era::LATENT_DIM});
    return emb;
}

}  // namespace

TEST(Evaluate, CosineSimMatrixNormalized) {
    const torch::Tensor z =
        torch::tensor({{1.0, 0.0}, {0.0, 2.0}, {1.0, 1.0}}, torch::kFloat64);

    const torch::Tensor sim = era::cosine_sim_matrix(z);
    EXPECT_EQ(sim.sizes(), (std::vector<int64_t>{3, 3}));
    EXPECT_TRUE(torch::allclose(sim.diagonal(), torch::ones({3}, torch::kFloat64), 1e-6, 1e-6));
    EXPECT_GE(sim.min().item<double>(), -1.0);
    EXPECT_LE(sim.max().item<double>(), 1.0);
}

TEST(Evaluate, QuerySimilarExcludesSamePlayer) {
    const era::Embeddings emb = fake_embeddings();

    const std::vector<era::SimilarMatch> matches = era::query_similar(emb, "P0", "2000-01", 5);
    EXPECT_EQ(matches.size(), 5u);
    for (const era::SimilarMatch& match : matches) {
        EXPECT_NE(match.player_name, "P0");
    }
}

TEST(Evaluate, EmbeddingsCsvRoundTrip) {
    constexpr int n = 12;
    constexpr int64_t input_dim = 8;
    torch::manual_seed(0);

    era::Frame df;
    for (int i = 0; i < n; ++i) {
        df.player_id.push_back(200 + i);
        // Row 0 carries a comma and a quote so the escape/split pair is exercised.
        df.player_name.push_back(i == 0 ? "Smith, Jr. \"Bo\"" : "P" + std::to_string(i));
        df.season_year.push_back(std::to_string(1990 + i) + "-" + std::to_string(91 + i));
        df.era.push_back(i % 2 == 0 ? "1990s" : "2020s");
        df.num["value_target"].push_back(i * 0.3333333333333333);
    }

    era::EraTranslatorDANN model(input_dim);
    const torch::Tensor X = torch::randn({n, input_dim});
    const std::filesystem::path out_path =
        std::filesystem::temp_directory_path() / "era_translator_emb_test.csv";

    const era::Embeddings written = era::export_embeddings(df, model, X, out_path.string());
    const era::Embeddings loaded = era::read_embeddings_csv(out_path.string());

    EXPECT_EQ(loaded.size(), written.size());
    EXPECT_EQ(loaded.player_id, written.player_id);
    EXPECT_EQ(loaded.player_name, written.player_name);
    EXPECT_EQ(loaded.season_year, written.season_year);
    EXPECT_EQ(loaded.era, written.era);
    EXPECT_EQ(loaded.value_target, written.value_target);
    for (int i = 0; i < n; ++i) {
        // value_pred is written at float precision, so it round-trips as a float.
        EXPECT_FLOAT_EQ(static_cast<float>(loaded.value_pred[i]),
                        static_cast<float>(written.value_pred[i]));
    }
    EXPECT_EQ(loaded.z.sizes(), written.z.sizes());
    EXPECT_TRUE(torch::equal(loaded.z, written.z));

    std::filesystem::remove(out_path);
}

TEST(Evaluate, WriteComparisonsExcludesSamePlayer) {
    const era::Embeddings emb = fake_embeddings();
    const std::filesystem::path out_path =
        std::filesystem::temp_directory_path() / "era_translator_cmp_test.csv";

    const std::vector<era::ComparisonRow> rows = era::write_comparisons(emb, out_path.string());
    EXPECT_GT(rows.size(), 0u);
    for (const era::ComparisonRow& row : rows) {
        EXPECT_NE(row.query_player, row.match_player);
    }
    std::set<std::string> eras;
    for (const era::ComparisonRow& row : rows) {
        eras.insert(row.match_era);
    }
    EXPECT_GE(eras.size(), 1u);

    std::filesystem::remove(out_path);
}
