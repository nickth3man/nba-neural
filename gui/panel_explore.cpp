// Explore panel: browse every player-season and pull its cross-era analogues.
#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

#include "app_state.hpp"
#include "misc/cpp/imgui_stdlib.h"
#include "panels.hpp"

namespace gui {
namespace {

enum Column { kColPlayer, kColSeason, kColEra, kColTarget, kColPred };

struct ExploreState {
    // Identity of the embedding set the row list was built from; a new pointer
    // means a different run was loaded and the list must be rebuilt.
    const era::Embeddings* source = nullptr;
    std::string name_filter;
    int era_choice = 0;  // 0 = all decades
    int top_k = 5;

    std::vector<int> rows;  // indices into the embeddings, filtered and sorted
    bool rebuild = true;

    int selected = -1;  // index into the embeddings
    std::string selected_label;
    std::vector<era::SimilarMatch> matches;
    std::string match_error;
};

ExploreState g_state;

bool contains_ci(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) {
        return true;
    }
    const auto it = std::search(haystack.begin(), haystack.end(), needle.begin(), needle.end(),
                                [](unsigned char a, unsigned char b) {
                                    return std::tolower(a) == std::tolower(b);
                                });
    return it != haystack.end();
}

void rebuild_rows(const era::Embeddings& emb) {
    g_state.rows.clear();
    g_state.rows.reserve(static_cast<std::size_t>(emb.size()));
    const std::string era_wanted =
        g_state.era_choice > 0 ? era::ERA_ORDER[static_cast<std::size_t>(g_state.era_choice - 1)]
                               : std::string();
    for (int i = 0; i < static_cast<int>(emb.size()); ++i) {
        if (!era_wanted.empty() && emb.era[static_cast<std::size_t>(i)] != era_wanted) {
            continue;
        }
        if (!contains_ci(emb.player_name[static_cast<std::size_t>(i)], g_state.name_filter)) {
            continue;
        }
        g_state.rows.push_back(i);
    }
}

void sort_rows(const era::Embeddings& emb, const ImGuiTableSortSpecs* specs) {
    if (specs == nullptr || specs->SpecsCount == 0) {
        return;
    }
    const ImGuiTableColumnSortSpecs& spec = specs->Specs[0];
    const bool ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
    const auto key_less = [&](int a, int b) {
        const auto ia = static_cast<std::size_t>(a);
        const auto ib = static_cast<std::size_t>(b);
        switch (spec.ColumnUserID) {
            case kColSeason:
                return emb.season_year[ia] < emb.season_year[ib];
            case kColEra:
                return emb.era[ia] < emb.era[ib];
            case kColTarget:
                return emb.value_target[ia] < emb.value_target[ib];
            case kColPred:
                return emb.value_pred[ia] < emb.value_pred[ib];
            case kColPlayer:
            default:
                return emb.player_name[ia] < emb.player_name[ib];
        }
    };
    std::stable_sort(g_state.rows.begin(), g_state.rows.end(),
                     [&](int a, int b) { return ascending ? key_less(a, b) : key_less(b, a); });
}

void select_row(const era::Embeddings& emb, int row) {
    g_state.selected = row;
    const auto i = static_cast<std::size_t>(row);
    g_state.selected_label = emb.player_name[i] + "  " + emb.season_year[i];
    g_state.match_error.clear();
    try {
        g_state.matches =
            era::query_similar(emb, emb.player_name[i], emb.season_year[i], g_state.top_k);
    } catch (const std::exception& error) {
        g_state.matches.clear();
        g_state.match_error = error.what();
    }
}

void draw_matches() {
    if (g_state.selected < 0) {
        ImGui::TextDisabled("Select a player-season to see its cross-era analogues.");
        return;
    }
    ImGui::Text("Most similar to %s", g_state.selected_label.c_str());
    if (!g_state.match_error.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "%s", g_state.match_error.c_str());
        return;
    }
    if (ImGui::BeginTable("matches", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                              ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("player");
        ImGui::TableSetupColumn("season");
        ImGui::TableSetupColumn("era");
        ImGui::TableSetupColumn("value");
        ImGui::TableSetupColumn("similarity");
        ImGui::TableHeadersRow();
        for (const era::SimilarMatch& match : g_state.matches) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(match.player_name.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(match.season_year.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(match.era.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", match.value_target);
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", match.cosine_sim);
        }
        ImGui::EndTable();
    }
}

}  // namespace

void draw_explore_panel(AppState& state) {
    const std::shared_ptr<const era::Embeddings> emb = state.embeddings();
    if (!emb) {
        ImGui::TextDisabled(
            "No embeddings loaded. Start a run, or load a finished run's embeddings.csv "
            "from the Run panel.");
        return;
    }
    if (g_state.source != emb.get()) {
        g_state.source = emb.get();
        g_state.rebuild = true;
        g_state.selected = -1;
        g_state.matches.clear();
    }

    ImGui::SetNextItemWidth(220);
    if (ImGui::InputText("player", &g_state.name_filter)) {
        g_state.rebuild = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    std::vector<const char*> era_items = {"All decades"};
    for (const std::string& name : era::ERA_ORDER) {
        era_items.push_back(name.c_str());
    }
    if (ImGui::Combo("era", &g_state.era_choice, era_items.data(),
                     static_cast<int>(era_items.size()))) {
        g_state.rebuild = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    if (ImGui::SliderInt("top k", &g_state.top_k, 1, 20) && g_state.selected >= 0) {
        select_row(*emb, g_state.selected);
    }

    if (g_state.rebuild) {
        rebuild_rows(*emb);
        g_state.rebuild = false;
    }
    ImGui::Text("%d of %lld player-seasons", static_cast<int>(g_state.rows.size()),
                static_cast<long long>(emb->size()));

    const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
                                  ImGuiTableFlags_SizingStretchProp;
    const float table_height = ImGui::GetContentRegionAvail().y * 0.6f;
    if (ImGui::BeginTable("seasons", 5, flags, ImVec2(0, table_height))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("player", ImGuiTableColumnFlags_DefaultSort, 0.0f, kColPlayer);
        ImGui::TableSetupColumn("season", 0, 0.0f, kColSeason);
        ImGui::TableSetupColumn("era", 0, 0.0f, kColEra);
        ImGui::TableSetupColumn("value target", 0, 0.0f, kColTarget);
        ImGui::TableSetupColumn("value pred", 0, 0.0f, kColPred);
        ImGui::TableHeadersRow();

        if (ImGuiTableSortSpecs* specs = ImGui::TableGetSortSpecs()) {
            if (specs->SpecsDirty) {
                sort_rows(*emb, specs);
                specs->SpecsDirty = false;
            }
        }

        // 23,516 rows: only the visible slice is ever built.
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(g_state.rows.size()));
        while (clipper.Step()) {
            for (int line = clipper.DisplayStart; line < clipper.DisplayEnd; ++line) {
                const int row = g_state.rows[static_cast<std::size_t>(line)];
                const auto i = static_cast<std::size_t>(row);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::PushID(row);
                if (ImGui::Selectable(emb->player_name[i].c_str(), g_state.selected == row,
                                      ImGuiSelectableFlags_SpanAllColumns)) {
                    select_row(*emb, row);
                }
                ImGui::PopID();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(emb->season_year[i].c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(emb->era[i].c_str());
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", emb->value_target[i]);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", emb->value_pred[i]);
            }
        }
        ImGui::EndTable();
    }

    ImGui::SeparatorText("Cross-era analogues");
    draw_matches();
}

}  // namespace gui
