// Latent panel: the decade clouds should overlap if the encoder is era-invariant.
#include <imgui.h>
#include <implot.h>
#include <implot3d.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "panels.hpp"

namespace gui {
namespace {

// One contiguous xyz block per decade, so ImPlot3D can take plain pointers.
struct DecadeCloud {
    std::string label;
    std::vector<float> x, y, z;
};

struct LatentState {
    const era::Embeddings* source = nullptr;
    int dim_x = 0, dim_y = 1, dim_z = 2;
    std::vector<DecadeCloud> clouds;
    bool rebuild = true;
};

LatentState g_state;

void rebuild_clouds(const era::Embeddings& emb) {
    g_state.clouds.clear();
    const int64_t latent_dim = emb.z.size(1);
    const int dx = std::clamp(g_state.dim_x, 0, static_cast<int>(latent_dim) - 1);
    const int dy = std::clamp(g_state.dim_y, 0, static_cast<int>(latent_dim) - 1);
    const int dz = std::clamp(g_state.dim_z, 0, static_cast<int>(latent_dim) - 1);

    const torch::Tensor z = emb.z.to(torch::kCPU).contiguous();
    const auto acc = z.accessor<float, 2>();

    // Keep ERA_ORDER's ordering so the legend colors stay stable across runs.
    for (const std::string& decade : era::ERA_ORDER) {
        DecadeCloud cloud;
        cloud.label = decade;
        for (int64_t i = 0; i < emb.size(); ++i) {
            if (emb.era[static_cast<std::size_t>(i)] != decade) {
                continue;
            }
            cloud.x.push_back(acc[i][dx]);
            cloud.y.push_back(acc[i][dy]);
            cloud.z.push_back(acc[i][dz]);
        }
        if (!cloud.x.empty()) {
            g_state.clouds.push_back(std::move(cloud));
        }
    }
}

void draw_dim_selector(const char* label, int* dim, int64_t latent_dim) {
    ImGui::SetNextItemWidth(90);
    if (ImGui::SliderInt(label, dim, 0, static_cast<int>(latent_dim) - 1)) {
        g_state.rebuild = true;
    }
}

void draw_tsne(AppState& state) {
    const Job& job = state.tsne_job();
    ImGui::BeginDisabled(job.running());
    if (ImGui::Button("Compute t-SNE")) {
        state.start_tsne();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!job.running());
    if (ImGui::Button("Cancel t-SNE")) {
        state.cancel_tsne();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (job.running()) {
        ImGui::TextUnformatted("running - exact t-SNE over 4,000 rows takes minutes");
    } else if (job.state() == JobState::Failed) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "error: %s", job.error().c_str());
    } else {
        ImGui::TextDisabled("no gnuplot needed");
    }

    const std::shared_ptr<const TsneResult> result = state.tsne_result();
    if (!result) {
        return;
    }
    const torch::Tensor xy = result->xy;
    const auto acc = xy.accessor<float, 2>();
    if (ImPlot::BeginPlot("##tsne", ImVec2(-1, 320), ImPlotFlags_Equal)) {
        ImPlot::SetupAxes("t-SNE 1", "t-SNE 2");
        for (const std::string& decade : era::ERA_ORDER) {
            std::vector<float> xs, ys;
            for (int64_t i = 0; i < xy.size(0); ++i) {
                if (result->era[static_cast<std::size_t>(i)] == decade) {
                    xs.push_back(acc[i][0]);
                    ys.push_back(acc[i][1]);
                }
            }
            if (!xs.empty()) {
                ImPlot::PlotScatter(decade.c_str(), xs.data(), ys.data(),
                                    static_cast<int>(xs.size()));
            }
        }
        ImPlot::EndPlot();
    }
}

}  // namespace

void draw_latent_panel(AppState& state) {
    const std::shared_ptr<const era::Embeddings> emb = state.embeddings();
    if (!emb) {
        ImGui::TextDisabled("No embeddings loaded.");
        return;
    }
    if (g_state.source != emb.get()) {
        g_state.source = emb.get();
        g_state.rebuild = true;
    }

    const int64_t latent_dim = emb->z.size(1);
    ImGui::TextWrapped(
        "Each decade is one series. If the encoder is era-invariant the clouds sit on top of "
        "one another; a decade that separates out is one the latent still encodes.");
    draw_dim_selector("x dim", &g_state.dim_x, latent_dim);
    ImGui::SameLine();
    draw_dim_selector("y dim", &g_state.dim_y, latent_dim);
    ImGui::SameLine();
    draw_dim_selector("z dim", &g_state.dim_z, latent_dim);

    if (g_state.rebuild) {
        rebuild_clouds(*emb);
        g_state.rebuild = false;
    }

    if (ImPlot3D::BeginPlot("##latent", ImVec2(-1, 380))) {
        ImPlot3D::SetupAxes("z_x", "z_y", "z_z");
        for (const DecadeCloud& cloud : g_state.clouds) {
            ImPlot3D::PlotScatter(cloud.label.c_str(), cloud.x.data(), cloud.y.data(),
                                  cloud.z.data(), static_cast<int>(cloud.x.size()));
        }
        ImPlot3D::EndPlot();
    }

    ImGui::SeparatorText("t-SNE");
    draw_tsne(state);
}

}  // namespace gui
