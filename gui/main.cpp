// Entry point for the Era Translator cockpit.
#include <implot.h>
#include <implot3d.h>

#include "app_state.hpp"
#include "hello_imgui/hello_imgui.h"
#include "panels.hpp"

int main(int, char**) {
    gui::AppState state;

    HelloImGui::RunnerParams params;
    params.appWindowParams.windowTitle = "Era Translator";
    params.appWindowParams.windowGeometry.size = {1500, 950};
    params.imGuiWindowParams.defaultImGuiWindowType =
        HelloImGui::DefaultImGuiWindowType::ProvideFullScreenDockSpace;
    params.imGuiWindowParams.showMenuBar = true;
    params.imGuiWindowParams.showStatusBar = true;
    // Otherwise the saved layout lands in the working directory, i.e. the repo.
    params.iniFolderType = HelloImGui::IniFolderType::AppUserConfigFolder;

    params.callbacks.PostInit = [] {
        ImPlot::CreateContext();
        ImPlot3D::CreateContext();
    };
    params.callbacks.BeforeExit = [] {
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
    };
    // Runs before the dockable windows, so finished worker threads are joined
    // once per frame no matter which panels are visible.
    params.callbacks.ShowGui = [&state] { state.poll(); };

    // Run is left, everything else shares a tabbed node on the right.
    params.dockingParams.dockingSplits = {
        {"MainDockSpace", "RightSpace", ImGuiDir_Right, 0.62f},
    };
    params.dockingParams.dockableWindows = {
        {"Run", "MainDockSpace", [&state] { gui::draw_run_panel(state); }},
        {"Explore", "RightSpace", [&state] { gui::draw_explore_panel(state); }},
        {"Latent", "RightSpace", [&state] { gui::draw_latent_panel(state); }},
        {"Metrics", "RightSpace", [&state] { gui::draw_metrics_panel(state); }},
    };

    HelloImGui::Run(params);
    return 0;
}
