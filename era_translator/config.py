"""Central configuration for the Era Translator pipeline."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "nba.duckdb"
DEFAULT_OUT = REPO_ROOT / "outputs" / "era_translator"

# Data extraction
SEASON_TYPE = "Regular"
MIN_GAMES = 10

# Feature columns (avg_pie is 100% NULL in this warehouse -> excluded).
# The six constituent stats of the value target (pts/reb/ast/stl/blk/tov) are
# deliberately EXCLUDED from features: if present, the value head copies the
# target linearly and R2 ~0.98 is meaningless (bottleneck fix).
COUNTING_FEATURES = ["avg_min", "gp"]
PCT_FEATURES = ["fg_pct", "fg3_pct", "ft_pct"]
ADVANCED_FEATURES = ["avg_net_rating", "avg_ts_pct", "avg_usg_pct"]
NUMERIC_FEATURES = COUNTING_FEATURES + PCT_FEATURES + ADVANCED_FEATURES

# Value composite (target): stats recorded in all eras contribute; stl/blk/tov are
# added only where recorded (0 before 1973-74 / 1977-78). Per-season z-score applied later.
VALUE_ALWAYS = ["avg_pts", "avg_reb", "avg_ast"]
VALUE_CONDITIONAL = {"avg_stl": 1.0, "avg_blk": 1.0, "avg_tov": -1.0}

# Era buckets
ERA_ORDER = ["1940s", "1950s", "1960s", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s"]

# Model architecture
ENCODER_HIDDEN = 128
LATENT_DIM = 24
ERA_HEAD_STEPS = 1
VALUE_HEAD_HIDDEN = 32
ERA_HEAD_HIDDEN = 128
ERA_HEAD_LR = 1e-2
DROPOUT = 0.2
NUM_ERAS = len(ERA_ORDER)

# Training
SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
# SGD + momentum: Rangwani et al. ICML 2022 show SGD reaches smoother task-loss
# minima than Adam, which stabilizes adversarial training (Adam caused era-head
# collapse). Ganin-style LR decay: mu_p = LR / (1 + 10*p)^0.75, p = epoch/MAX_EPOCHS.
LR = 1e-2
LR_ALPHA = 10.0
LR_BETA = 0.75
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 150
PATIENCE = 15
# Early stopping is disabled until epoch 30 so the lambda schedule reaches >= 0.9
# before patience can fire.
PATIENCE_WARMUP_EPOCHS = 30
LAMBDA_MAX = 2.0
LAMBDA_GAMMA = 10.0
CENTROID_ALIGN_WEIGHT = 5.0
# Per-element clip on the reversed gradient inside the GRL (PaddleSpeech pattern;
# GRADA identifies exploding discriminator gradients as the key failure mode).
GRL_GRAD_CLIP = 0.5
