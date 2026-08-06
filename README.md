# Era Translator

A domain-adversarial neural network (DANN) that learns **era-invariant player-season
embeddings** from 80 seasons of NBA data, so cross-era player comparisons are fair.

The problem: raw box-score stats are not comparable across eras. A 1962 Wilt
Chamberlain season (50.4 PPG in a 130-PPG league) looks like an outlier today,
not because he was better than a modern star, but because the game was played
differently. The Era Translator trains an encoder that maps a player-season
statistical profile into a latent space that is:

1. **predictive** of the player's within-era value (a per-season z-scored box-score
   composite), and
2. **uninformative about era** — a fresh logistic-regression classifier cannot
   guess the decade from the embedding much better than chance.

The result: embeddings you can compare directly across decades. Query LeBron's
2023-24 season and get Karl Malone, Kawhi Leonard, Dirk Nowitzki, and Julius
Erving; query Wilt's 1961-62 and get Kareem, Walt Bellamy, and Wes Unseld.

## How it works

```
player-season stats (23,516 rows, 1946-47 .. 2025-26)
        |
        v
  Encoder (MLP -> 24-dim latent, linear output)
        |                         |
        |                  Era head (128-128-9)
        |                  with Gradient Reversal Layer
        v                         |
  Value head (-> z-score)   +  centroid-alignment loss
```

- **Data**: `agg_player_season` joined to `dim_player` (age, position), regular
  season only, `gp >= 10`. All DuckDB access is read-only.
- **Features**: minutes, games, shooting percentages, TS%, usage, net rating,
  age, position. The six constituent stats of the value target
  (pts/reb/ast/stl/blk/tov) are deliberately **excluded** from the features —
  otherwise the value head would copy the target linearly and R² would be
  meaningless.
- **Value target**: raw box-score composite, z-scored within each season so it is
  era-normalized by construction.
- **Era removal**:
  - Gradient reversal (Ganin et al., arXiv:1505.07818) with a sigmoid lambda
    schedule; the era head trains on era-balanced batches.
  - **Centroid alignment** (CAT-style, arXiv:2407.12782): pulls each decade's
    latent centroid toward the global centroid — the mechanism that actually
    strips era, since a collapsed discriminator's reversal gradient is useless.
  - SGD + momentum (Rangwani et al., ICML 2022: SGD reaches smoother task-loss
    minima than Adam, which stabilizes adversarial training).
- **Similarity**: correlation (centered cosine) similarity. Plain cosine collapses
  toward 0.99 for all pairs once the latent is era-aligned; centering removes the
  shared direction and restores player discrimination.

## Setup

Requires Python >= 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # installs pinned-latest deps into .venv (CPU torch)
uv run pytest    # 13 tests
uv run ruff check era_translator tests
uv run ty check era_translator
```

The warehouse lives at `data/nba.duckdb` (read-only source; not committed).

## Run

```bash
uv run python -m era_translator.main --db data/nba.duckdb --out outputs/era_translator
```

This runs extraction -> feature engineering -> training -> evaluation and writes:

| Artifact | Contents |
|---|---|
| `outputs/era_translator/model.pt` | Trained DANN checkpoint |
| `outputs/era_translator/embeddings.csv` | 23,516 player-seasons x `z_0..z_23` + value target/prediction |
| `outputs/era_translator/tsne_era.png` | t-SNE of the latent colored by decade |
| `outputs/era_translator/comparisons.csv` | For every 2020s player-season, top-5 similar seasons in each other decade |
| `outputs/era_translator/metrics.json` | Test R²/MSE, fresh-LR era probe, per-decade accuracy |

## Results (final run)

| Metric | Value |
|---|---|
| Test value R² | 0.944 |
| Fresh LR era probe (balanced) | 0.160 (random = 0.111; raw features = 0.376) |
| Embeddings | 23,516 rows, 24-dim latent |

**Era probe**: a fresh logistic-regression classifier trained on train-split
embeddings scores 0.160 balanced accuracy on held-out embeddings — down from
0.376 on the raw features, approaching the 0.111 random baseline. The in-training
era head is *not* used for this check: it collapses to predicting a single decade
once the adversarial loss dominates (balanced accuracy exactly 1/9), which is a
degenerate equilibrium, not evidence of invariance.

## Project layout

```
era_translator/
  config.py      # hyperparameters, feature lists, era buckets
  data.py        # read-only DuckDB extraction (aggregates multi-team stints)
  features.py    # imputation, age/position, value target, splits
  model.py       # DANN: encoder + value head + era head + GRL
  train.py       # SGD + momentum, era-balanced batches, centroid alignment
  evaluate.py    # metrics, embeddings export, t-SNE, comparisons, era probe
  main.py        # CLI entry point
tests/           # pytest suite (13 tests)
```

## Known limitations

- The 1940s (smallest, most distinctive decade: no 3PT, imputed stats) retains
  some separability in the latent. The next lever is per-era feature normalization
  or removing era-marker features such as `fg3_pct` from the input.
- The value target is a within-season z-score; "value" therefore means relative
  production within a season, not an absolute all-time scale.
