"""Read-only extraction of player-season data from the DuckDB warehouse."""

import duckdb
import pandas as pd

from .config import DEFAULT_DB, MIN_GAMES, SEASON_TYPE


def _load_con(db_path):
    return duckdb.connect(str(db_path), read_only=True)


def load_player_seasons(db_path=DEFAULT_DB, min_games=MIN_GAMES) -> pd.DataFrame:
    """Load one row per (player_id, season_year) for regular seasons.

    Handles three warehouse quirks discovered during exploration:
      1. agg_player_season has one row per team stint -> aggregate to one
         row per player-season, weighting per-game stats by games played.
      2. dim_player has SCD-style duplicate rows per player -> keep the row
         marked is_current, else the latest valid_from.
      3. Only 3 positions (G/F/C) plus NULL -> NULL becomes "Unknown".

    All reads are read-only (read_only=True).
    """
    con = _load_con(db_path)
    try:
        df = con.execute(
            f"""
            WITH player_agg AS (
                SELECT
                    player_id,
                    season_year,
                    SUM(gp) AS gp,
                    SUM(total_min) AS total_min,
                    SUM(total_pts) AS total_pts,
                    SUM(total_reb) AS total_reb,
                    SUM(total_ast) AS total_ast,
                    SUM(total_stl) AS total_stl,
                    SUM(total_blk) AS total_blk,
                    SUM(total_tov) AS total_tov,
                    SUM(total_fgm) AS total_fgm,
                    SUM(total_fga) AS total_fga,
                    SUM(total_fg3m) AS total_fg3m,
                    SUM(total_fg3a) AS total_fg3a,
                    SUM(total_ftm) AS total_ftm,
                    SUM(total_fta) AS total_fta,
                    SUM(avg_off_rating * gp) AS off_rating_w,
                    SUM(avg_def_rating * gp) AS def_rating_w,
                    SUM(avg_net_rating * gp) AS net_rating_w,
                    SUM(avg_ts_pct * gp) AS ts_pct_w,
                    SUM(avg_usg_pct * gp) AS usg_pct_w,
                    SUM(avg_pie * gp) AS pie_w
                FROM agg_player_season
                WHERE season_type = '{SEASON_TYPE}'
                GROUP BY player_id, season_year
                HAVING SUM(gp) >= {min_games}
            ),
            player_dedup AS (
                SELECT player_id, full_name, birth_date, position
                FROM (
                    SELECT player_id, full_name, birth_date, position,
                           ROW_NUMBER() OVER (
                               PARTITION BY player_id
                               ORDER BY is_current DESC, valid_from DESC
                           ) AS rn
                    FROM dim_player
                )
                WHERE rn = 1
            )
            SELECT
                p.player_id,
                p.season_year,
                p.gp,
                p.total_min / NULLIF(p.gp, 0) AS avg_min,
                p.total_pts / NULLIF(p.gp, 0) AS avg_pts,
                p.total_reb / NULLIF(p.gp, 0) AS avg_reb,
                p.total_ast / NULLIF(p.gp, 0) AS avg_ast,
                p.total_stl / NULLIF(p.gp, 0) AS avg_stl,
                p.total_blk / NULLIF(p.gp, 0) AS avg_blk,
                p.total_tov / NULLIF(p.gp, 0) AS avg_tov,
                CASE WHEN p.total_fga > 0 THEN p.total_fgm / p.total_fga END AS fg_pct,
                CASE WHEN p.total_fg3a > 0 THEN p.total_fg3m / p.total_fg3a END AS fg3_pct,
                CASE WHEN p.total_fta > 0 THEN p.total_ftm / p.total_fta END AS ft_pct,
                p.off_rating_w / NULLIF(p.gp, 0) AS avg_off_rating,
                p.def_rating_w / NULLIF(p.gp, 0) AS avg_def_rating,
                p.net_rating_w / NULLIF(p.gp, 0) AS avg_net_rating,
                p.ts_pct_w / NULLIF(p.gp, 0) AS avg_ts_pct,
                p.usg_pct_w / NULLIF(p.gp, 0) AS avg_usg_pct,
                p.pie_w / NULLIF(p.gp, 0) AS avg_pie,
                d.full_name AS player_name,
                d.birth_date,
                COALESCE(d.position, 'Unknown') AS position
            FROM player_agg p
            LEFT JOIN player_dedup d USING (player_id)
            ORDER BY p.season_year, p.player_id
            """
        ).fetchdf()
    finally:
        con.close()
    return df
