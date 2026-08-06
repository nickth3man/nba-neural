"""Evaluation: metrics, embedding export, t-SNE plot, cross-era comparisons."""

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from .config import ERA_ORDER, LATENT_DIM

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_model(model, X, y, era, device="cpu"):
    """Return (mse, r2, era_acc, era_bal_acc, per_decade_acc) on the given split."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        _, vpred, epred = model(Xt)
        vpred = vpred.cpu().numpy()
        epred = epred.cpu().numpy()
    mse = float(np.mean((vpred - y) ** 2))
    ss_res = np.sum((vpred - y) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    era_pred = epred.argmax(1)
    era_acc = float(np.mean(era_pred == era))
    per_decade = {}
    recalls = []
    for i, dec in enumerate(ERA_ORDER):
        mask = era == i
        if mask.sum() > 0:
            r = float(np.mean(era_pred[mask] == i))
            per_decade[dec] = r
            recalls.append(r)
    # Balanced accuracy: macro-averaged per-decade recall. Collapse to a single
    # class yields 1/9 ~ 0.111, same as true random guessing.
    era_bal_acc = float(np.mean(recalls)) if recalls else 0.0
    return mse, r2, era_acc, era_bal_acc, per_decade


def embed_all(model, X, device="cpu"):
    """Return latent embeddings (N, LATENT_DIM) for all rows."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        z, _, _ = model(Xt)
        return z.cpu().numpy()


def export_embeddings(df, scaler, model, X, out_path, device="cpu"):
    """Write embeddings.csv with identity columns, target, prediction, and z_0..z_63."""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        z, vpred, _ = model(Xt)
        z = z.cpu().numpy()
        vpred = vpred.cpu().numpy()
    out = df[["player_id", "player_name", "season_year", "era", "value_target"]].copy()
    out["value_pred"] = vpred
    for d in range(LATENT_DIM):
        out[f"z_{d}"] = z[:, d]
    out.to_csv(out_path, index=False)
    return out


def plot_tsne(embeddings_df, out_path, sample=4000, seed=42):
    """2-D t-SNE colored by era; decades should overlap if the space is invariant."""
    rng = np.random.RandomState(seed)
    if len(embeddings_df) > sample:
        idx = rng.choice(len(embeddings_df), sample, replace=False)
        sub = embeddings_df.iloc[idx]
    else:
        sub = embeddings_df
    zcols = [f"z_{d}" for d in range(LATENT_DIM)]
    tsne = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(sub) - 1))
    xy = tsne.fit_transform(sub[zcols].values)
    era_colors = {
        "1940s": "#8c564b", "1950s": "#e377c2", "1960s": "#7f7f7f",
        "1970s": "#bcbd22", "1980s": "#17becf", "1990s": "#1f77b4",
        "2000s": "#ff7f0e", "2010s": "#2ca02c", "2020s": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(10, 8))
    for era in ERA_ORDER:
        mask = sub["era"].values == era
        ax.scatter(xy[mask, 0], xy[mask, 1], s=6, alpha=0.5,
                   color=era_colors[era], label=era)
    ax.set_title("t-SNE of Era-Translator latent space (colored by decade)")
    ax.legend(markerscale=3, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def cosine_sim_matrix(z):
    """Correlation similarity: centered cosine (Pearson-like).

    Plain cosine collapses toward ~0.99 for all pairs once the centroid-alignment
    loss pulls the latent toward a shared direction (observed mean 0.987), which
    destroys player discrimination. Centering removes the shared direction; the
    resulting sims span [-1, 1] and recover meaningful cross-era matches.
    """
    zc = z - z.mean(0, keepdims=True)
    zn = zc / (np.linalg.norm(zc, axis=1, keepdims=True) + 1e-12)
    return zn @ zn.T


def write_comparisons(embeddings_df, out_path, top_k=5, seed=42):
    """For each 2020s player-season, top-k most similar across every other decade.

    Excludes the query player's own other seasons (same player_id) so matches are
    *other* players, ranked by cosine similarity within each era.
    """
    zcols = [f"z_{d}" for d in range(LATENT_DIM)]
    z = embeddings_df[zcols].values
    sim = cosine_sim_matrix(z)
    rows = []
    modern = embeddings_df[embeddings_df["era"] == "2020s"].index
    pid = embeddings_df["player_id"].values
    for i in modern:
        for era in ERA_ORDER:
            if era == "2020s":
                continue
            mask = (embeddings_df["era"].values == era) & (pid != pid[i])
            scores = np.where(mask, sim[i], -np.inf)
            order = np.argsort(-scores)[:top_k]
            for j in order:
                if not np.isfinite(scores[j]):
                    continue
                rows.append({
                    "query_player": embeddings_df.loc[i, "player_name"],
                    "query_season": embeddings_df.loc[i, "season_year"],
                    "query_value": round(embeddings_df.loc[i, "value_target"], 3),
                    "match_era": era,
                    "match_player": embeddings_df.loc[j, "player_name"],
                    "match_season": embeddings_df.loc[j, "season_year"],
                    "match_value": round(embeddings_df.loc[j, "value_target"], 3),
                    "cosine_sim": round(float(sim[i][j]), 4),
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return pd.DataFrame(rows)


def era_probe_balanced(embeddings_df, seed=42):
    """Fresh logistic-regression era probe on frozen embeddings.

    The in-training era head collapses to predict-a-single-decade (balanced acc
    exactly 1/9) once the adversarial loss dominates, so it is NOT a valid
    invariance measure. A fresh LR on held-out embeddings is the honest check:
    balanced accuracy near 1/9 means era info is genuinely stripped from z.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import train_test_split

    zcols = [f"z_{d}" for d in range(LATENT_DIM)]
    Z = embeddings_df[zcols].values
    y = embeddings_df["era"].map(
        {e: i for i, e in enumerate(ERA_ORDER)}
    ).values
    Ztr, Zte, ytr, yte = train_test_split(
        Z, y, test_size=0.3, stratify=y, random_state=seed
    )
    lr = LogisticRegression(max_iter=2000).fit(Ztr, ytr)
    return float(balanced_accuracy_score(yte, lr.predict(Zte)))


def query_similar(embeddings_df, player_name, season_year, top_k=5):
    """Return top-k most similar player-seasons across all eras.

    Excludes the query player's own other seasons (same player_id).
    """
    qmask = (
        embeddings_df["player_name"].eq(player_name)
        & embeddings_df["season_year"].eq(season_year)
    )
    if qmask.sum() == 0:
        raise ValueError(f"No embedding for {player_name} {season_year}")
    i = int(np.where(qmask.values)[0][0])
    zcols = [f"z_{d}" for d in range(LATENT_DIM)]
    z = embeddings_df[zcols].values
    sim = cosine_sim_matrix(z)
    qid = embeddings_df.loc[i, "player_id"]
    scores = np.where(embeddings_df["player_id"].values != qid, sim[i], -np.inf)
    order = np.argsort(-scores)
    keep = order[:top_k]
    return embeddings_df.iloc[keep][
        ["player_name", "season_year", "era", "value_target", "value_pred"]
    ].assign(cosine_sim=scores[keep])
