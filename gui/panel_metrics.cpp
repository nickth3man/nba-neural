// Metrics panel: the numbers metrics.json carries, plus per-decade era recall.
#include <imgui.h>
#include <implot.h>

#include <optional>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "panels.hpp"

namespace gui {

void draw_metrics_panel(AppState& state) {
    const std::optional<era::Metrics> metrics = state.metrics();
    if (!metrics) {
        ImGui::TextDisabled("No metrics yet - start a run from the Run panel.");
        return;
    }

    ImGui::SeparatorText("Value head");
    ImGui::Text("test R2   %.4f", metrics->test_value_r2);
    ImGui::Text("test MSE  %.4f", metrics->test_value_mse);
    ImGui::Text("rows      %lld", static_cast<long long>(metrics->rows));
    ImGui::Text("epochs    %lld", static_cast<long long>(metrics->epochs_trained));

    ImGui::SeparatorText("Era invariance");
    ImGui::Text("fresh LR era probe (balanced)  %.4f", metrics->fresh_lr_era_probe_balanced);
    ImGui::Text("random baseline                %.4f", metrics->random_era_accuracy);
    ImGui::TextWrapped(
        "The probe is the honest check: a fresh classifier on held-out embeddings. "
        "Closer to the baseline means era is genuinely stripped from the latent. "
        "The in-training era head below collapses once the adversarial loss dominates, "
        "so its balanced accuracy of exactly 1/9 is a degenerate equilibrium.");
    ImGui::Text("in-training era head (balanced) %.4f", metrics->test_era_balanced_accuracy);

    ImGui::SeparatorText("Per-decade era recall");
    std::vector<double> values;
    std::vector<const char*> labels;
    std::vector<double> positions;
    values.reserve(metrics->per_decade_era_accuracy.size());
    for (std::size_t i = 0; i < metrics->per_decade_era_accuracy.size(); ++i) {
        values.push_back(metrics->per_decade_era_accuracy[i].second);
        labels.push_back(metrics->per_decade_era_accuracy[i].first.c_str());
        positions.push_back(static_cast<double>(i));
    }
    if (ImPlot::BeginPlot("##per_decade", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("decade", "recall");
        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), static_cast<int>(positions.size()),
                               labels.data());
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 1.0, ImPlotCond_Always);
        ImPlot::PlotBars("recall", values.data(), static_cast<int>(values.size()), 0.6);
        const double random_acc = metrics->random_era_accuracy;
        ImPlot::PlotInfLines("##baseline", &random_acc, 1,
                             {ImPlotProp_Flags, ImPlotInfLinesFlags_Horizontal});
        ImPlot::EndPlot();
    }
}

}  // namespace gui
