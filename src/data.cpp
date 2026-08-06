#include "data.hpp"

#include <duckdb.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace era {
namespace {

enum class ColType { Int64, Double, Varchar };

struct ColSpec {
    const char* name;
    ColType type;
};

// Column order must match the SELECT list below.
const std::vector<ColSpec> kColumns = {
    {"player_id", ColType::Int64},        {"season_year", ColType::Varchar},
    {"gp", ColType::Double},              {"avg_min", ColType::Double},
    {"avg_pts", ColType::Double},         {"avg_reb", ColType::Double},
    {"avg_ast", ColType::Double},         {"avg_stl", ColType::Double},
    {"avg_blk", ColType::Double},         {"avg_tov", ColType::Double},
    {"fg_pct", ColType::Double},          {"fg3_pct", ColType::Double},
    {"ft_pct", ColType::Double},          {"avg_off_rating", ColType::Double},
    {"avg_def_rating", ColType::Double},  {"avg_net_rating", ColType::Double},
    {"avg_ts_pct", ColType::Double},      {"avg_usg_pct", ColType::Double},
    {"avg_pie", ColType::Double},         {"player_name", ColType::Varchar},
    {"birth_date", ColType::Varchar},     {"position", ColType::Varchar},
};

std::string build_sql(int min_games) {
    // Same query as the Python original. Every projected column is cast so the
    // chunk reader below only has to handle BIGINT, DOUBLE and VARCHAR.
    // birth_date stays a VARCHAR rather than being year-extracted in SQL so
    // compute_age() keeps parsing dates itself and stays independently testable.
    return R"(
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
                WHERE season_type = ')" +
           std::string(SEASON_TYPE) + R"('
                GROUP BY player_id, season_year
                HAVING SUM(gp) >= )" +
           std::to_string(min_games) + R"(
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
                CAST(p.player_id AS BIGINT) AS player_id,
                CAST(p.season_year AS VARCHAR) AS season_year,
                CAST(p.gp AS DOUBLE) AS gp,
                CAST(p.total_min / NULLIF(p.gp, 0) AS DOUBLE) AS avg_min,
                CAST(p.total_pts / NULLIF(p.gp, 0) AS DOUBLE) AS avg_pts,
                CAST(p.total_reb / NULLIF(p.gp, 0) AS DOUBLE) AS avg_reb,
                CAST(p.total_ast / NULLIF(p.gp, 0) AS DOUBLE) AS avg_ast,
                CAST(p.total_stl / NULLIF(p.gp, 0) AS DOUBLE) AS avg_stl,
                CAST(p.total_blk / NULLIF(p.gp, 0) AS DOUBLE) AS avg_blk,
                CAST(p.total_tov / NULLIF(p.gp, 0) AS DOUBLE) AS avg_tov,
                CAST(CASE WHEN p.total_fga > 0 THEN p.total_fgm / p.total_fga END AS DOUBLE) AS fg_pct,
                CAST(CASE WHEN p.total_fg3a > 0 THEN p.total_fg3m / p.total_fg3a END AS DOUBLE) AS fg3_pct,
                CAST(CASE WHEN p.total_fta > 0 THEN p.total_ftm / p.total_fta END AS DOUBLE) AS ft_pct,
                CAST(p.off_rating_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_off_rating,
                CAST(p.def_rating_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_def_rating,
                CAST(p.net_rating_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_net_rating,
                CAST(p.ts_pct_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_ts_pct,
                CAST(p.usg_pct_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_usg_pct,
                CAST(p.pie_w / NULLIF(p.gp, 0) AS DOUBLE) AS avg_pie,
                CAST(d.full_name AS VARCHAR) AS player_name,
                CAST(d.birth_date AS VARCHAR) AS birth_date,
                CAST(COALESCE(d.position, 'Unknown') AS VARCHAR) AS position
            FROM player_agg p
            LEFT JOIN player_dedup d USING (player_id)
            ORDER BY p.season_year, p.player_id
            )";
}

// Frees the DuckDB handles even when a read throws part way through a chunk.
struct Connection {
    duckdb_database db = nullptr;
    duckdb_connection con = nullptr;

    explicit Connection(const std::string& path) {
        duckdb_config config = nullptr;
        if (duckdb_create_config(&config) == DuckDBError) {
            throw std::runtime_error("duckdb_create_config failed");
        }
        duckdb_set_config(config, "access_mode", "READ_ONLY");
        char* err = nullptr;
        const duckdb_state state = duckdb_open_ext(path.c_str(), &db, config, &err);
        duckdb_destroy_config(&config);
        if (state == DuckDBError) {
            const std::string message = err ? err : "unknown error";
            if (err) {
                duckdb_free(err);
            }
            throw std::runtime_error("cannot open " + path + ": " + message);
        }
        if (duckdb_connect(db, &con) == DuckDBError) {
            duckdb_close(&db);
            throw std::runtime_error("duckdb_connect failed");
        }
    }

    ~Connection() {
        if (con) {
            duckdb_disconnect(&con);
        }
        if (db) {
            duckdb_close(&db);
        }
    }

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;
};

std::string read_string(duckdb_vector vec, idx_t row) {
    auto* data = static_cast<duckdb_string_t*>(duckdb_vector_get_data(vec));
    return std::string(duckdb_string_t_data(&data[row]), duckdb_string_t_length(data[row]));
}

}  // namespace

Frame load_player_seasons(const std::string& db_path, int min_games) {
    Connection connection(db_path);

    duckdb_result result;
    if (duckdb_query(connection.con, build_sql(min_games).c_str(), &result) == DuckDBError) {
        const std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw std::runtime_error("query failed: " + message);
    }

    Frame frame;
    for (const ColSpec& col : kColumns) {
        if (col.type == ColType::Double) {
            frame.num[col.name] = {};
        }
    }

    while (true) {
        duckdb_data_chunk chunk = duckdb_fetch_chunk(result);
        if (!chunk) {
            break;
        }
        const idx_t rows = duckdb_data_chunk_get_size(chunk);
        for (idx_t c = 0; c < kColumns.size(); ++c) {
            const ColSpec& spec = kColumns[c];
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, c);
            uint64_t* validity = duckdb_vector_get_validity(vec);

            for (idx_t r = 0; r < rows; ++r) {
                const bool valid = !validity || duckdb_validity_row_is_valid(validity, r);
                switch (spec.type) {
                    case ColType::Int64: {
                        auto* data = static_cast<int64_t*>(duckdb_vector_get_data(vec));
                        frame.player_id.push_back(valid ? data[r] : 0);
                        break;
                    }
                    case ColType::Double: {
                        auto* data = static_cast<double*>(duckdb_vector_get_data(vec));
                        // NULL becomes NaN so the imputation chain in features.cpp
                        // sees the same missingness pandas did.
                        frame.num[spec.name].push_back(valid ? data[r] : std::nan(""));
                        break;
                    }
                    case ColType::Varchar: {
                        std::string value = valid ? read_string(vec, r) : std::string();
                        const std::string name = spec.name;
                        if (name == "season_year") {
                            frame.season_year.push_back(std::move(value));
                        } else if (name == "player_name") {
                            frame.player_name.push_back(std::move(value));
                        } else if (name == "birth_date") {
                            frame.birth_date.push_back(std::move(value));
                        } else {
                            frame.position.push_back(std::move(value));
                        }
                        break;
                    }
                }
            }
        }
        duckdb_destroy_data_chunk(&chunk);
    }
    duckdb_destroy_result(&result);
    return frame;
}

}  // namespace era
