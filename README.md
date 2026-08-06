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
2023-24 season and get Karl Malone, Julius Erving, Patrick Ewing, and Kevin Durant;
query Wilt's 1961-62 and get his cross-era peers rather than his contemporaries.

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

Requires a C++20 compiler (MSVC 2022+), CMake 3.24+, and Ninja. The build targets
Windows x64; `duckdb.dll` and the torch DLLs are copied next to the executables.

CMake fetches its dependencies on first configure:

| Dependency | How it arrives |
|---|---|
| **LibTorch** 2.13.0+cpu | Downloaded (~250 MB), or reused via `-DCMAKE_PREFIX_PATH` |
| **DuckDB** 1.5.5 | Prebuilt `libduckdb-windows-amd64.zip` (~13 MB), no compile |
| **Matplot++** 1.2.2 | Built from source |
| **GoogleTest** 1.17.0 | Built from source |
| **Hello ImGui** 1.92.700 (+ Dear ImGui, GLFW) | Built from source (GUI only) |
| **ImPlot** 1.0 / **ImPlot3D** 0.4 | Built from source (GUI only) |

Every dependency is permissively licensed — BSD-3 (LibTorch, GoogleTest), MIT
(DuckDB, Matplot++, Hello ImGui, Dear ImGui, ImPlot, ImPlot3D) or zlib (GLFW).
Pass `-DBUILD_GUI=OFF` to skip the GUI dependencies and build only the CLI.

DuckDB is used through its **C API** (`duckdb.h`) — the project's own docs describe
the C++ API as internal and unstable and recommend the C API for applications.

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

If you already have a libtorch on disk (a PyTorch install ships one), point at it
to skip the download:

```bash
cmake -B build -G Ninja -DCMAKE_PREFIX_PATH=<path>/site-packages/torch
```

Then build and test:

```bash
cmake --build build
```

```bash
ctest --test-dir build --output-on-failure
```

**gnuplot is a runtime prerequisite for the t-SNE plot only.** Matplot++ renders
through it, so without gnuplot 5.2.6+ on PATH every stage still runs and the plot
step is skipped with a message. Configure warns if it is missing.

```bash
scoop install gnuplot
```

The warehouse lives at `data/nba.duckdb` (read-only source; not committed).

## Run

```bash
./build/era_translator --db data/nba.duckdb --out outputs/era_translator
```

This runs extraction -> feature engineering -> training -> evaluation and writes the
table below. On CPU the pipeline takes ~35 seconds, plus ~3 minutes for the t-SNE
plot (exact t-SNE over a 4,000-row sample); without gnuplot it finishes in the 35.

| Artifact | Contents |
|---|---|
| `outputs/era_translator/model.pt` | Trained DANN checkpoint (LibTorch archive) |
| `outputs/era_translator/embeddings.csv` | 23,516 player-seasons x `z_0..z_23` + value target/prediction |
| `outputs/era_translator/tsne_era.png` | t-SNE of the latent colored by decade (needs gnuplot) |
| `outputs/era_translator/comparisons.csv` | For every 2020s player-season, top-5 similar seasons in each other decade |
| `outputs/era_translator/metrics.json` | Test R²/MSE, fresh-LR era probe, per-decade accuracy |

Flags: `--db`, `--out`, `--epochs`, `--device`.

## GUI

`era_translator_gui` is a desktop cockpit over the same pipeline. It links
`era_translator_lib` directly — no subprocess, no stdout parsing — so training
runs in-process on a worker thread and reports every epoch as it happens.

```bash
./build/era_translator_gui
```

| Panel | What it shows |
|---|---|
| **Run** | Config, start/cancel, and live curves: train/val loss, val era accuracy against the 1/9 random line, plus the λ and LR schedules with a marker at the epoch early stopping arms |
| **Explore** | All 23,516 player-seasons, filterable and sortable, with the top-k cross-era analogues of the selected row (`era::query_similar`) |
| **Latent** | 3-D scatter of any three latent dimensions, one series per decade — they should overlap — and a 2-D t-SNE rendered on demand |
| **Metrics** | R², MSE, the fresh-LR era probe against its baseline, and per-decade era recall |

**Load a finished run** from the Run panel to explore without retraining: point it
at an `embeddings.csv` and it picks up the sibling `metrics.json` too. Loading
the full 23,516 rows takes ~150 ms.

**The GUI does not need gnuplot.** It plots through ImPlot, including the t-SNE,
so gnuplot remains a CLI-only prerequisite. The GUI writes the same artifacts as
the CLI except `tsne_era.png`.

## Results (final run)

| Metric | Value |
|---|---|
| Test value R² | 0.939 |
| Test value MSE | 0.061 |
| Fresh LR era probe (balanced) | 0.169 (random = 0.111) |
| Embeddings | 23,516 rows, 24-dim latent |

**Era probe**: a fresh logistic-regression classifier trained on train-split
embeddings scores 0.169 balanced accuracy on held-out embeddings, approaching the
0.111 random baseline. (The same probe on the raw input features scored 0.376 in
the original Python implementation.) The in-training era head is *not* used for
this check: it collapses to predicting a single decade once the adversarial loss
dominates (balanced accuracy exactly 1/9), which is a degenerate equilibrium, not
evidence of invariance.

## Project layout

```
CMakeLists.txt   # FetchContent deps, DLL staging, gnuplot check
src/
  config.hpp     # hyperparameters, feature lists, era buckets
  frame.hpp      # column-oriented table used in place of a DataFrame
  data.cpp       # read-only DuckDB extraction (aggregates multi-team stints)
  features.cpp   # imputation, age/position, value target, splits, scaling
  model.cpp      # DANN: encoder + value head + era head + GRL
  train.cpp      # SGD + momentum, era-balanced batches, centroid alignment
  evaluate.cpp   # metrics, embeddings export, t-SNE, comparisons, era probe
  io.cpp         # CSV escaping, metrics JSON
  main.cpp       # CLI entry point
gui/
  app_state.cpp  # background jobs (pipeline, training, t-SNE) + UI snapshots
  panel_run.cpp  # config, start/cancel, live loss and schedule plots
  panel_explore.cpp  # player-season table + cross-era analogues
  panel_latent.cpp   # 3-D latent scatter by decade, 2-D t-SNE
  panel_metrics.cpp  # metrics readout + per-decade recall
  main.cpp       # Hello ImGui runner and docking layout
tests/           # GoogleTest suite (14 tests)
```

## Known limitations

- The 1940s (smallest, most distinctive decade: no 3PT, imputed stats) retains
  some separability in the latent, and its matches score far lower similarity than
  every other decade. The next lever is per-era feature normalization or removing
  era-marker features such as `fg3_pct` from the input.
- The value target is a within-season z-score; "value" therefore means relative
  production within a season, not an absolute all-time scale.
- t-SNE and the logistic-regression era probe are implemented directly on LibTorch
  autograd rather than wrapping scikit-learn, so their numbers are close to but not
  identical to the Python originals. The train/val/test split is likewise a
  stratified shuffle driven by `std::mt19937_64`, not numpy's RNG, so it selects
  different rows than the Python did.
