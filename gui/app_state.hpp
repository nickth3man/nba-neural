// Shared state for the cockpit: background jobs and the data the panels draw.
//
// Every long operation runs on a worker thread; the UI thread only ever reads
// snapshots taken under `mutex_`. Embedding and t-SNE results are handed over as
// shared_ptr so a panel can hold one for the duration of a frame while a worker
// swaps in a newer one.
#pragma once

#include <torch/torch.h>

#include <atomic>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "evaluate.hpp"
#include "io.hpp"
#include "train.hpp"

namespace gui {

enum class JobState { Idle, Running, Done, Failed, Cancelled };

// One cancellable background thread. The worker is handed the cancel flag and
// is expected to poll it between stages.
class Job {
   public:
    ~Job();

    void start(std::function<void(const std::atomic<bool>&)> work);
    void cancel() { cancel_ = true; }

    bool running() const { return state_.load() == JobState::Running; }
    JobState state() const { return state_.load(); }
    // Only meaningful once state() is Failed.
    std::string error() const;

    // Called once per frame from the UI thread so a finished thread is joined
    // rather than left dangling until the next start().
    void reap();

   private:
    std::thread thread_;
    std::atomic<bool> cancel_{false};
    std::atomic<JobState> state_{JobState::Idle};
    std::atomic<bool> finished_{false};
    mutable std::mutex error_mutex_;
    std::string error_;
};

struct RunConfig {
    std::string db = "data/nba.duckdb";
    std::string out = "outputs/gui";
    int epochs = era::MAX_EPOCHS;
    bool use_cuda = false;
};

// Column-per-series so ImPlot can take raw pointers without repacking.
struct TrainingCurves {
    std::vector<double> epoch;
    std::vector<double> train_loss;
    std::vector<double> val_loss;
    std::vector<double> val_era_acc;
    std::vector<double> lambda;
    std::vector<double> lr;
    int best_epoch = -1;

    std::size_t size() const { return epoch.size(); }
    void clear();
};

// 2-D t-SNE of a latent sample, with the decade of each point for coloring.
struct TsneResult {
    torch::Tensor xy;  // (M, 2) float32 CPU
    std::vector<std::string> era;
};

class AppState {
   public:
    RunConfig config;

    // --- background work -------------------------------------------------
    // Full pipeline: extract -> features -> train -> evaluate -> artifacts.
    void start_run();
    void cancel_run() { pipeline_.cancel(); }
    const Job& pipeline() const { return pipeline_; }

    // Load a finished run's embeddings.csv without retraining.
    void load_embeddings(const std::string& path);

    // Exact t-SNE over a sample; minutes, so it is its own cancellable job.
    void start_tsne();
    void cancel_tsne() { tsne_.cancel(); }
    const Job& tsne_job() const { return tsne_; }

    // Joins finished threads. Call once per frame from the UI thread.
    void poll();

    // --- snapshots for the UI thread -------------------------------------
    std::shared_ptr<const era::Embeddings> embeddings() const;
    std::shared_ptr<const TsneResult> tsne_result() const;
    std::optional<era::Metrics> metrics() const;
    TrainingCurves curves() const;
    std::vector<std::string> log() const;
    std::string stage() const;

    void log_line(std::string line);

   private:
    void set_stage(std::string stage);
    void run_pipeline(const std::atomic<bool>& cancel);

    Job pipeline_;
    Job tsne_;

    mutable std::mutex mutex_;
    std::shared_ptr<const era::Embeddings> embeddings_;
    std::shared_ptr<const TsneResult> tsne_result_;
    std::optional<era::Metrics> metrics_;
    TrainingCurves curves_;
    std::deque<std::string> log_;
    std::string stage_;
};

}  // namespace gui
