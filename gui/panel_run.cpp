// Run panel: launch the pipeline and watch the adversarial dynamics live.
#include <imgui.h>
#include <implot.h>
#include <torch/torch.h>

#include <string>
#include <vector>

#include "app_state.hpp"
#include "config.hpp"
#include "misc/cpp/imgui_stdlib.h"
#include "panels.hpp"

namespace gui {
namespace {

std::string load_path = "outputs/cpp/embeddings.csv";

const char* job_status(const Job& job) {
    switch (job.state()) {
        case JobState::Running:
            return "running";
        case JobState::Done:
            return "done";
        case JobState::Failed:
            return "failed";
        case JobState::Cancelled:
            return "cancelled";
        case JobState::Idle:
        default:
            return "idle";
    }
}

void draw_curves(const TrainingCurves& curves) {
    const int count = static_cast<int>(curves.size());

    if (ImPlot::BeginPlot("Loss and era accuracy", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("epoch", "loss");
        ImPlot::SetupAxis(ImAxis_Y2, "val era accuracy", ImPlotAxisFlags_AuxDefault);
        ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0, 1.0, ImPlotCond_Always);
        if (count > 0) {
            ImPlot::PlotLine("train loss", curves.epoch.data(), curves.train_loss.data(), count);
            ImPlot::PlotLine("val loss", curves.epoch.data(), curves.val_loss.data(), count);

            ImPlot::SetAxis(ImAxis_Y2);
            ImPlot::PlotLine("val era acc", curves.epoch.data(), curves.val_era_acc.data(), count);
            // A collapsed era head sits exactly on 1/9: era accuracy hugging this
            // line is degenerate equilibrium, not evidence of invariance.
            const double random_acc = 1.0 / static_cast<double>(era::ERA_ORDER.size());
            const double xs[2] = {curves.epoch.front(), curves.epoch.back()};
            const double ys[2] = {random_acc, random_acc};
            ImPlot::PlotLine("random (1/9)", xs, ys, 2);
        }
        ImPlot::EndPlot();
    }

    if (ImPlot::BeginPlot("Schedules", ImVec2(-1, 200))) {
        ImPlot::SetupAxes("epoch", "lambda");
        ImPlot::SetupAxis(ImAxis_Y2, "lr", ImPlotAxisFlags_AuxDefault);
        if (count > 0) {
            ImPlot::PlotLine("lambda", curves.epoch.data(), curves.lambda.data(), count);
            // Early stopping is inert until this epoch, so the lambda ramp can finish.
            const double warmup = static_cast<double>(era::PATIENCE_WARMUP_EPOCHS);
            ImPlot::PlotInfLines("patience arms", &warmup, 1);

            ImPlot::SetAxis(ImAxis_Y2);
            ImPlot::PlotLine("lr", curves.epoch.data(), curves.lr.data(), count);
        }
        ImPlot::EndPlot();
    }
}

}  // namespace

void draw_run_panel(AppState& state) {
    const Job& pipeline = state.pipeline();
    const bool busy = pipeline.running();

    ImGui::SeparatorText("Configuration");
    ImGui::BeginDisabled(busy);
    ImGui::InputText("DuckDB warehouse", &state.config.db);
    ImGui::InputText("Output directory", &state.config.out);
    ImGui::InputInt("Max epochs", &state.config.epochs);
    if (state.config.epochs < 1) {
        state.config.epochs = 1;
    }
    const bool cuda = torch::cuda::is_available();
    ImGui::BeginDisabled(!cuda);
    ImGui::Checkbox(cuda ? "Use CUDA" : "Use CUDA (unavailable)", &state.config.use_cuda);
    ImGui::EndDisabled();
    ImGui::EndDisabled();

    ImGui::SeparatorText("Run");
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Start run")) {
        state.start_run();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!busy);
    if (ImGui::Button("Cancel")) {
        state.cancel_run();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextUnformatted(job_status(pipeline));

    const std::string stage = state.stage();
    if (!stage.empty()) {
        ImGui::TextUnformatted(stage.c_str());
    }
    if (pipeline.state() == JobState::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "error: %s", pipeline.error().c_str());
    }

    ImGui::SeparatorText("Load a finished run");
    ImGui::BeginDisabled(busy);
    ImGui::InputText("embeddings.csv", &load_path);
    if (ImGui::Button("Load")) {
        state.load_embeddings(load_path);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("explore a previous run without retraining");

    ImGui::SeparatorText("Training");
    const TrainingCurves curves = state.curves();
    if (curves.size() == 0) {
        ImGui::TextDisabled("No epochs yet.");
    } else {
        ImGui::Text("epoch %d/%d   best epoch %d", static_cast<int>(curves.epoch.back()),
                    state.config.epochs - 1, curves.best_epoch);
        ImGui::ProgressBar(static_cast<float>(curves.size()) /
                           static_cast<float>(state.config.epochs));
    }
    draw_curves(curves);

    ImGui::SeparatorText("Log");
    if (ImGui::BeginChild("log", ImVec2(0, 140), ImGuiChildFlags_Borders)) {
        for (const std::string& line : state.log()) {
            ImGui::TextUnformatted(line.c_str());
        }
        // Stick to the newest line while a run is producing output.
        if (busy && ImGui::GetScrollY() >= ImGui::GetScrollMaxY()) {
            ImGui::SetScrollHereY(1.0f);
        }
    }
    ImGui::EndChild();
}

}  // namespace gui
