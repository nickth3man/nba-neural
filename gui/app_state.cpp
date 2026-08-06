#include "app_state.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <utility>

#include "config.hpp"
#include "data.hpp"
#include "features.hpp"
#include "model.hpp"

namespace gui {
namespace {

constexpr std::size_t kMaxLogLines = 500;

// metrics.json is written by era::write_metrics_json and nothing else, so its
// shape is fixed and a targeted scan beats pulling in a JSON dependency.
// Returns nullopt when a key is missing, i.e. the file is not one of ours.
std::optional<double> json_number(const std::string& text, const std::string& key) {
    const std::size_t at = text.find('"' + key + '"');
    if (at == std::string::npos) {
        return std::nullopt;
    }
    const std::size_t colon = text.find(':', at);
    if (colon == std::string::npos) {
        return std::nullopt;
    }
    try {
        return std::stod(text.substr(colon + 1));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<era::Metrics> read_metrics_json(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::optional<double> rows = json_number(text, "rows");
    if (!rows) {
        return std::nullopt;
    }
    era::Metrics m;
    m.rows = static_cast<int64_t>(*rows);
    m.test_value_mse = json_number(text, "test_value_mse").value_or(0.0);
    m.test_value_r2 = json_number(text, "test_value_r2").value_or(0.0);
    m.test_era_accuracy = json_number(text, "test_era_accuracy").value_or(0.0);
    m.test_era_balanced_accuracy = json_number(text, "test_era_balanced_accuracy").value_or(0.0);
    m.fresh_lr_era_probe_balanced = json_number(text, "fresh_lr_era_probe_balanced").value_or(0.0);
    m.random_era_accuracy = json_number(text, "random_era_accuracy").value_or(0.0);
    m.epochs_trained = static_cast<int64_t>(json_number(text, "epochs_trained").value_or(0.0));

    // The decade keys live in their own object; scan only that slice so the
    // outer "test_era_accuracy" keys cannot be mistaken for one.
    const std::size_t block = text.find("\"per_decade_era_accuracy\"");
    if (block != std::string::npos) {
        const std::size_t open = text.find('{', block);
        const std::size_t close = text.find('}', open);
        if (open != std::string::npos && close != std::string::npos) {
            const std::string slice = text.substr(open, close - open);
            for (const std::string& decade : era::ERA_ORDER) {
                if (const std::optional<double> value = json_number(slice, decade)) {
                    m.per_decade_era_accuracy.emplace_back(decade, *value);
                }
            }
        }
    }
    return m;
}

// Same std::vector -> tensor conversions main.cpp does; features.hpp is
// deliberately torch-free, so the two-line helpers live at each call site.
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

}  // namespace

// ------------------------------------------------------------------- Job

Job::~Job() {
    cancel_ = true;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void Job::start(std::function<void(const std::atomic<bool>&)> work) {
    if (running()) {
        return;
    }
    if (thread_.joinable()) {
        thread_.join();
    }
    cancel_ = false;
    finished_ = false;
    state_ = JobState::Running;
    thread_ = std::thread([this, work = std::move(work)]() {
        JobState outcome = JobState::Done;
        try {
            work(cancel_);
            if (cancel_) {
                outcome = JobState::Cancelled;
            }
        } catch (const std::exception& error) {
            {
                const std::lock_guard<std::mutex> guard(error_mutex_);
                error_ = error.what();
            }
            outcome = JobState::Failed;
        }
        state_ = outcome;
        finished_ = true;
    });
}

std::string Job::error() const {
    const std::lock_guard<std::mutex> guard(error_mutex_);
    return error_;
}

void Job::reap() {
    if (finished_.exchange(false) && thread_.joinable()) {
        thread_.join();
    }
}

// --------------------------------------------------------- TrainingCurves

void TrainingCurves::clear() {
    epoch.clear();
    train_loss.clear();
    val_loss.clear();
    val_era_acc.clear();
    lambda.clear();
    lr.clear();
    best_epoch = -1;
}

// --------------------------------------------------------------- AppState

void AppState::log_line(std::string line) {
    const std::lock_guard<std::mutex> guard(mutex_);
    log_.push_back(std::move(line));
    while (log_.size() > kMaxLogLines) {
        log_.pop_front();
    }
}

void AppState::set_stage(std::string stage) {
    const std::lock_guard<std::mutex> guard(mutex_);
    stage_ = std::move(stage);
}

std::shared_ptr<const era::Embeddings> AppState::embeddings() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return embeddings_;
}

std::shared_ptr<const TsneResult> AppState::tsne_result() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return tsne_result_;
}

std::optional<era::Metrics> AppState::metrics() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return metrics_;
}

TrainingCurves AppState::curves() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return curves_;
}

std::vector<std::string> AppState::log() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return std::vector<std::string>(log_.begin(), log_.end());
}

std::string AppState::stage() const {
    const std::lock_guard<std::mutex> guard(mutex_);
    return stage_;
}

void AppState::poll() {
    pipeline_.reap();
    tsne_.reap();
}

void AppState::load_embeddings(const std::string& path) {
    if (pipeline_.running()) {
        return;
    }
    pipeline_.start([this, path](const std::atomic<bool>&) {
        set_stage("Loading " + path);
        log_line("Loading " + path);
        auto loaded = std::make_shared<era::Embeddings>(era::read_embeddings_csv(path));
        log_line("Loaded " + std::to_string(loaded->size()) + " embeddings, latent dim " +
                 std::to_string(loaded->z.size(1)));

        // The run's metrics.json sits beside its embeddings.csv; pick it up so
        // the Metrics panel works for a loaded run too, not just a fresh one.
        const std::filesystem::path sibling =
            std::filesystem::path(path).parent_path() / "metrics.json";
        std::optional<era::Metrics> loaded_metrics = read_metrics_json(sibling.generic_string());
        if (loaded_metrics) {
            log_line("Loaded metrics from " + sibling.generic_string());
        }

        {
            const std::lock_guard<std::mutex> guard(mutex_);
            embeddings_ = std::move(loaded);
            metrics_ = std::move(loaded_metrics);
            tsne_result_.reset();  // belongs to the previous embedding set
        }
        set_stage("");
    });
}

void AppState::start_tsne() {
    const std::shared_ptr<const era::Embeddings> emb = embeddings();
    if (!emb || tsne_.running()) {
        return;
    }
    tsne_.start([this, emb](const std::atomic<bool>& cancel) {
        // Same 4,000-row sample the CLI plot uses; exact t-SNE is O(N^2).
        constexpr int64_t kSample = 4000;
        const int64_t n = emb->size();
        const int64_t take = std::min<int64_t>(kSample, n);
        log_line("t-SNE on " + std::to_string(take) + " sampled rows (this takes minutes)...");

        torch::manual_seed(era::SEED);
        const torch::Tensor pick =
            take < n ? torch::randperm(n, torch::TensorOptions().dtype(torch::kInt64)).slice(0, 0, take)
                     : torch::arange(n, torch::TensorOptions().dtype(torch::kInt64));
        if (cancel) {
            return;
        }

        auto result = std::make_shared<TsneResult>();
        result->xy = era::tsne(emb->z.index_select(0, pick), era::SEED).to(torch::kCPU).contiguous();
        if (cancel) {
            return;
        }
        result->era.reserve(static_cast<std::size_t>(take));
        for (int64_t i = 0; i < take; ++i) {
            result->era.push_back(emb->era[static_cast<std::size_t>(pick[i].item<int64_t>())]);
        }
        {
            const std::lock_guard<std::mutex> guard(mutex_);
            tsne_result_ = std::move(result);
        }
        log_line("t-SNE done.");
    });
}

void AppState::start_run() {
    if (pipeline_.running()) {
        return;
    }
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        curves_.clear();
        metrics_.reset();
    }
    pipeline_.start([this](const std::atomic<bool>& cancel) { run_pipeline(cancel); });
}

void AppState::run_pipeline(const std::atomic<bool>& cancel) {
    namespace fs = std::filesystem;
    const RunConfig cfg = config;  // copied so the UI can keep editing the form
    const fs::path out_dir(cfg.out);
    fs::create_directories(out_dir);
    const torch::Device device(cfg.use_cuda ? torch::kCUDA : torch::kCPU);

    torch::manual_seed(era::SEED);

    set_stage("Extracting player-seasons");
    log_line("Extracting from " + cfg.db + " (read-only)...");
    const era::Frame raw = era::load_player_seasons(cfg.db);
    log_line("  " + std::to_string(raw.size()) + " player-seasons");
    if (cancel) {
        return;
    }

    set_stage("Engineering features");
    const era::Frame df = era::engineer_features(raw);
    const std::vector<std::string> cols = era::feature_columns(df);
    const std::vector<int64_t> era_labels = era::era_codes(df);
    if (cancel) {
        return;
    }

    set_stage("Splitting and scaling");
    const era::Splits splits = era::make_splits(df);
    const era::Scaler scaler = era::fit_scaler(df, splits.train, cols);
    const era::Matrix features = era::transform_features(df, scaler, cols);
    log_line("  train=" + std::to_string(splits.train.size()) +
             " val=" + std::to_string(splits.val.size()) +
             " test=" + std::to_string(splits.test.size()));

    const torch::Tensor X = float_tensor(features.data, features.rows, features.cols);
    std::vector<float> targets(df.size());
    for (std::size_t i = 0; i < df.size(); ++i) {
        targets[i] = static_cast<float>(df.num.at("value_target")[i]);
    }
    const torch::Tensor y =
        float_tensor(targets, static_cast<int64_t>(targets.size()), 1).squeeze(1);
    const torch::Tensor era_tensor = index_tensor(era_labels);
    const torch::Tensor train_idx = index_tensor(splits.train);
    const torch::Tensor val_idx = index_tensor(splits.val);
    const torch::Tensor test_idx = index_tensor(splits.test);
    if (cancel) {
        return;
    }

    set_stage("Training");
    era::EraTranslatorDANN model(static_cast<int64_t>(cols.size()));
    const era::TrainResult trained = era::train_model(
        model, X.index_select(0, train_idx), y.index_select(0, train_idx),
        era_tensor.index_select(0, train_idx), X.index_select(0, val_idx),
        y.index_select(0, val_idx), era_tensor.index_select(0, val_idx), cfg.epochs, device,
        [this, &cancel](const era::EpochUpdate& update) {
            const std::lock_guard<std::mutex> guard(mutex_);
            curves_.epoch.push_back(update.epoch);
            curves_.train_loss.push_back(update.train_loss);
            curves_.val_loss.push_back(update.val_loss);
            curves_.val_era_acc.push_back(update.val_era_acc);
            curves_.lambda.push_back(update.lambda);
            curves_.lr.push_back(update.lr);
            if (update.improved) {
                curves_.best_epoch = update.epoch;
            }
            return !cancel;
        });
    if (cancel) {
        log_line("Cancelled during training.");
        return;
    }
    log_line("Trained " + std::to_string(trained.history.val_loss.size()) + " epochs.");

    set_stage("Saving model");
    const std::string model_path = (out_dir / "model.pt").generic_string();
    {
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.write("input_dim", torch::tensor(static_cast<int64_t>(cols.size())));
        archive.write("num_eras", torch::tensor(static_cast<int64_t>(era::ERA_ORDER.size())));
        archive.save_to(model_path);
    }

    set_stage("Evaluating");
    const era::EvalResult evaluation =
        era::evaluate_model(model, X.index_select(0, test_idx), y.index_select(0, test_idx),
                            era_tensor.index_select(0, test_idx), device);
    if (cancel) {
        return;
    }

    set_stage("Exporting embeddings");
    auto emb = std::make_shared<era::Embeddings>(era::export_embeddings(
        df, model, X, (out_dir / "embeddings.csv").generic_string(), device));
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        embeddings_ = emb;
        tsne_result_.reset();
    }
    log_line("Embeddings: " + std::to_string(emb->size()) + " rows");
    if (cancel) {
        return;
    }

    set_stage("Era probe");
    const double era_probe = era::era_probe_balanced(*emb);
    if (cancel) {
        return;
    }

    // The CLI plots t-SNE here through gnuplot; the GUI renders it on demand
    // from era::tsne() instead, so no plot is written.
    set_stage("Writing comparisons");
    era::write_comparisons(*emb, (out_dir / "comparisons.csv").generic_string());

    set_stage("Writing metrics");
    era::Metrics m;
    m.rows = static_cast<int64_t>(df.size());
    m.test_value_mse = evaluation.mse;
    m.test_value_r2 = evaluation.r2;
    m.test_era_accuracy = evaluation.era_acc;
    m.test_era_balanced_accuracy = evaluation.era_bal_acc;
    m.fresh_lr_era_probe_balanced = era_probe;
    m.random_era_accuracy = 1.0 / static_cast<double>(era::ERA_ORDER.size());
    m.per_decade_era_accuracy = evaluation.per_decade;
    m.epochs_trained = static_cast<int64_t>(trained.history.val_loss.size());
    era::write_metrics_json((out_dir / "metrics.json").generic_string(), m);
    {
        const std::lock_guard<std::mutex> guard(mutex_);
        metrics_ = m;
    }

    log_line("Run complete -> " + out_dir.generic_string());
    set_stage("");
}

}  // namespace gui
