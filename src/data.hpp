// Read-only extraction of player-season data from the DuckDB warehouse.
#pragma once

#include <string>

#include "config.hpp"
#include "frame.hpp"

namespace era {

// Load one row per (player_id, season_year) for regular seasons.
//
// Handles three warehouse quirks discovered during exploration:
//   1. agg_player_season has one row per team stint -> aggregate to one
//      row per player-season, weighting per-game stats by games played.
//   2. dim_player has SCD-style duplicate rows per player -> keep the row
//      marked is_current, else the latest valid_from.
//   3. Only 3 positions (G/F/C) plus NULL -> NULL becomes "Unknown".
//
// All reads are read-only (access_mode=READ_ONLY).
Frame load_player_seasons(const std::string& db_path, int min_games = MIN_GAMES);

}  // namespace era
