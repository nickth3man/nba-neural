// Central configuration for the Era Translator pipeline.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace era {

// Data extraction
inline constexpr const char* SEASON_TYPE = "Regular";
inline constexpr int MIN_GAMES = 10;

// Feature columns (avg_pie is 100% NULL in this warehouse -> excluded).
// The six constituent stats of the value target (pts/reb/ast/stl/blk/tov) are
// deliberately EXCLUDED from features: if present, the value head copies the
// target linearly and R2 ~0.98 is meaningless (bottleneck fix).
inline const std::vector<std::string> COUNTING_FEATURES = {"avg_min", "gp"};
inline const std::vector<std::string> PCT_FEATURES = {"fg_pct", "fg3_pct", "ft_pct"};
inline const std::vector<std::string> ADVANCED_FEATURES = {"avg_net_rating", "avg_ts_pct",
                                                           "avg_usg_pct"};
inline const std::vector<std::string> NUMERIC_FEATURES = {
    "avg_min", "gp", "fg_pct", "fg3_pct", "ft_pct", "avg_net_rating", "avg_ts_pct", "avg_usg_pct"};

// Value composite (target): stats recorded in all eras contribute; stl/blk/tov are
// added only where recorded (0 before 1973-74 / 1977-78). Per-season z-score applied later.
inline const std::vector<std::string> VALUE_ALWAYS = {"avg_pts", "avg_reb", "avg_ast"};
inline const std::vector<std::pair<std::string, double>> VALUE_CONDITIONAL = {
    {"avg_stl", 1.0}, {"avg_blk", 1.0}, {"avg_tov", -1.0}};

// Era buckets
inline const std::vector<std::string> ERA_ORDER = {"1940s", "1950s", "1960s", "1970s", "1980s",
                                                   "1990s", "2000s", "2010s", "2020s"};

inline const std::vector<std::string> POSITIONS = {"G", "F", "C", "Unknown"};

// Model architecture
inline constexpr int ENCODER_HIDDEN = 128;
inline constexpr int LATENT_DIM = 24;
inline constexpr int VALUE_HEAD_HIDDEN = 32;
inline constexpr int ERA_HEAD_HIDDEN = 128;
inline constexpr double ERA_HEAD_LR = 1e-2;
inline constexpr double DROPOUT = 0.2;
inline constexpr int NUM_ERAS = 9;

// Training
inline constexpr int64_t SEED = 42;
inline constexpr double TEST_FRAC = 0.15;
inline constexpr double VAL_FRAC = 0.15;
// SGD + momentum: Rangwani et al. ICML 2022 show SGD reaches smoother task-loss
// minima than Adam, which stabilizes adversarial training (Adam caused era-head
// collapse). Ganin-style LR decay: mu_p = LR / (1 + 10*p)^0.75, p = epoch/MAX_EPOCHS.
inline constexpr double LR = 1e-2;
inline constexpr double LR_ALPHA = 10.0;
inline constexpr double LR_BETA = 0.75;
inline constexpr double MOMENTUM = 0.9;
inline constexpr double WEIGHT_DECAY = 1e-3;
inline constexpr int BATCH_SIZE = 256;
inline constexpr int MAX_EPOCHS = 150;
inline constexpr int PATIENCE = 15;
// Early stopping is disabled until epoch 30 so the lambda schedule reaches >= 0.9
// before patience can fire.
inline constexpr int PATIENCE_WARMUP_EPOCHS = 30;
inline constexpr double LAMBDA_MAX = 2.0;
inline constexpr double LAMBDA_GAMMA = 10.0;
inline constexpr double CENTROID_ALIGN_WEIGHT = 5.0;

}  // namespace era
