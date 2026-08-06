// CLI entry point for the Era Translator pipeline.
#include <torch/torch.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.hpp"
#include "data.hpp"
#include "evaluate.hpp"
#include "features.hpp"
#include "io.hpp"
#include "model.hpp"
#include "train.hpp"

namespace {

std::string with_thousands(int64_t value) {
    std::string digits = std::to_string(value);
    for (int64_t pos = static_cast<int64_t>(digits.size()) - 3; pos > 0; pos -= 3) {
        digits.insert(static_cast<std::size_t>(pos), ",");
    }
    return digits;
}

torch::Tensor float_tensor(const std::vector<float>& values, int64_t rows, int64_t cols) {
    return torch::from_blob(const_cast<float*>(values.data()), {rows, cols},
                            torch::TensorOptions().dtype(torch::kFloat32))
        .clone();
}

torch::Tensor index_tensor(const std::vector<int64_t>& values) {
    return torch::from_blob(const_cast<int64_t*>(values.data()),
                            {static_cast<int64_t>(values.size())},
                            torch::TensorOptions().dtype(torch::kInt64))
        .clone();
}

bool gnuplot_available() {
#ifdef _WIN32
    return std::system("gnuplot --version > NUL 2>&1") == 0;
#else
    return std::system("gnuplot --version > /dev/null 2>&1") == 0;
#endif
}

struct Args {
    std::string db = "data/nba.duckdb";
    std::string out = "outputs/era_translator";
    int epochs = era::MAX_EPOCHS;
    std::string device = "cpu";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(flag + " requires a value");
            }
            return argv[++i];
        };
        if (flag == "--db") {
            args.db = value();
        } else if (flag == "--out") {
            args.out = value();
        } else if (flag == "--epochs") {
            args.epochs = std::stoi(value());
        } else if (flag == "--device") {
            args.device = value();
        } else if (flag == "-h" || flag == "--help") {
            std::cout << "Era Translator: DANN cross-era normalization\n"
                      << "  --db <path>      DuckDB warehouse (default data/nba.duckdb)\n"
                      << "  --out <dir>      output directory (default outputs/era_translator)\n"
                      << "  --epochs <n>     max training epochs (default "
                      << era::MAX_EPOCHS << ")\n"
                      << "  --device <dev>   cpu or cuda (default cpu)\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + flag);
        }
    }
    return args;
}

int run(const Args& args) {
    namespace fs = std::filesystem;
    const fs::path out_dir(args.out);
    fs::create_directories(out_dir);
    const torch::Device device(args.device);

    torch::manual_seed(era::SEED);

    std::cout << "[1/6] Extracting player-season data (read-only DuckDB)...\n";
    const era::Frame raw = era::load_player_seasons(args.db);
    const auto [min_season, max_season] =
        std::minmax_element(raw.season_year.begin(), raw.season_year.end());
    std::cout << "      " << with_thousands(static_cast<int64_t>(raw.size()))
              << " player-seasons, " << *min_season << " .. " << *max_season << "\n";

    std::cout << "[2/6] Engineering features...\n";
    const era::Frame df = era::engineer_features(raw);
    const std::vector<std::string> cols = era::feature_columns(df);
    const std::vector<int64_t> era_labels = era::era_codes(df);

    std::cout << "[3/6] Splitting (stratified by era)...\n";
    const era::Splits splits = era::make_splits(df);
    const era::Scaler scaler = era::fit_scaler(df, splits.train, cols);
    const era::Matrix features = era::transform_features(df, scaler, cols);

    const torch::Tensor X = float_tensor(features.data, features.rows, features.cols);
    std::vector<float> targets(df.size());
    for (std::size_t i = 0; i < df.size(); ++i) {
        targets[i] = static_cast<float>(df.num.at("value_target")[i]);
    }
    const torch::Tensor y = float_tensor(targets, static_cast<int64_t>(targets.size()), 1).squeeze(1);
    const torch::Tensor era_tensor = index_tensor(era_labels);

    const torch::Tensor train_idx = index_tensor(splits.train);
    const torch::Tensor val_idx = index_tensor(splits.val);
    const torch::Tensor test_idx = index_tensor(splits.test);
    std::cout << "      train=" << splits.train.size() << " val=" << splits.val.size()
              << " test=" << splits.test.size() << "\n";

    std::cout << "[4/6] Training DANN...\n";
    era::EraTranslatorDANN model(static_cast<int64_t>(cols.size()));
    const era::TrainResult trained = era::train_model(
        model, X.index_select(0, train_idx), y.index_select(0, train_idx),
        era_tensor.index_select(0, train_idx), X.index_select(0, val_idx),
        y.index_select(0, val_idx), era_tensor.index_select(0, val_idx), args.epochs, device);

    const std::string model_path = (out_dir / "model.pt").generic_string();
    {
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.write("input_dim", torch::tensor(static_cast<int64_t>(cols.size())));
        archive.write("num_eras", torch::tensor(static_cast<int64_t>(era::ERA_ORDER.size())));
        archive.save_to(model_path);
    }
    std::cout << "      saved " << model_path << "\n";

    std::cout << "[5/6] Evaluating on test set...\n";
    const era::EvalResult evaluation =
        era::evaluate_model(model, X.index_select(0, test_idx), y.index_select(0, test_idx),
                            era_tensor.index_select(0, test_idx), device);
    const double random_accuracy = 1.0 / static_cast<double>(era::ERA_ORDER.size());
    std::printf("      test value MSE %.4f | R2 %.4f\n", evaluation.mse, evaluation.r2);
    std::printf("      test era accuracy %.4f | balanced %.4f (random = %.4f)\n",
                evaluation.era_acc, evaluation.era_bal_acc, random_accuracy);
    for (const auto& [decade, accuracy] : evaluation.per_decade) {
        std::printf("        %s: %.4f\n", decade.c_str(), accuracy);
    }

    std::cout << "[6/6] Writing artifacts...\n";
    const era::Embeddings emb =
        era::export_embeddings(df, model, X, (out_dir / "embeddings.csv").generic_string(), device);
    std::cout << "      embeddings: " << with_thousands(emb.size()) << " rows -> "
              << (out_dir / "embeddings.csv").generic_string() << "\n";

    const double era_probe = era::era_probe_balanced(emb);
    std::printf("      fresh LR era probe (balanced): %.4f (random = %.4f)\n", era_probe,
                random_accuracy);

    const std::string plot_path = (out_dir / "tsne_era.png").generic_string();
    if (!gnuplot_available()) {
        std::cout << "      tsne plot SKIPPED: gnuplot was not found on PATH.\n"
                  << "        Matplot++ renders through gnuplot 5.2.6+.\n"
                  << "        Install with:  scoop install gnuplot   (or: choco install gnuplot)\n";
    } else {
        try {
            era::plot_tsne(emb, plot_path);
            std::cout << "      tsne plot -> " << plot_path << "\n";
        } catch (const std::exception& error) {
            // Keep the remaining artifacts: training has already happened and
            // the plot is the qualitative companion to metrics.json, not a gate.
            std::cout << "      tsne plot FAILED: " << error.what() << "\n"
                      << "        gnuplot 5.2.6+ with the pngcairo terminal is required.\n";
        }
    }

    const std::vector<era::ComparisonRow> comparisons =
        era::write_comparisons(emb, (out_dir / "comparisons.csv").generic_string());
    std::cout << "      comparisons: " << with_thousands(static_cast<int64_t>(comparisons.size()))
              << " rows -> " << (out_dir / "comparisons.csv").generic_string() << "\n";

    era::Metrics metrics;
    metrics.rows = static_cast<int64_t>(df.size());
    metrics.test_value_mse = evaluation.mse;
    metrics.test_value_r2 = evaluation.r2;
    metrics.test_era_accuracy = evaluation.era_acc;
    metrics.test_era_balanced_accuracy = evaluation.era_bal_acc;
    metrics.fresh_lr_era_probe_balanced = era_probe;
    metrics.random_era_accuracy = random_accuracy;
    metrics.per_decade_era_accuracy = evaluation.per_decade;
    metrics.epochs_trained = static_cast<int64_t>(trained.history.val_loss.size());
    era::write_metrics_json((out_dir / "metrics.json").generic_string(), metrics);
    std::cout << "      metrics -> " << (out_dir / "metrics.json").generic_string() << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_args(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
