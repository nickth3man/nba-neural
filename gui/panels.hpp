// The four cockpit panels. Each is drawn inside its own dockable window.
#pragma once

namespace gui {

class AppState;

void draw_run_panel(AppState& state);
void draw_metrics_panel(AppState& state);
void draw_explore_panel(AppState& state);
void draw_latent_panel(AppState& state);

}  // namespace gui
