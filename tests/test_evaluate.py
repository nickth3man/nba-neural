"""Unit tests for evaluation helpers."""

import numpy as np
import pandas as pd

from era_translator.config import LATENT_DIM
from era_translator.evaluate import (
    cosine_sim_matrix,
    query_similar,
    write_comparisons,
)


def _fake_embeddings(n=30):
    rng = np.random.RandomState(0)
    z = rng.randn(n, LATENT_DIM)
    rows = []
    for i in range(n):
        era = ["1940s", "1990s", "2020s"][i % 3]
        rows.append({
            "player_id": 100 + (i % 6),
            "player_name": f"P{i}",
            "season_year": f"{2000 + i}-01",
            "era": era,
            "value_target": float(i),
            "value_pred": float(i),
        })
    emb = pd.DataFrame(rows)
    for d in range(LATENT_DIM):
        emb[f"z_{d}"] = z[:, d]
    return emb


def test_cosine_sim_matrix_normalized():
    z = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    s = cosine_sim_matrix(z)
    assert s.shape == (3, 3)
    np.testing.assert_allclose(np.diag(s), np.ones(3), atol=1e-6)
    assert -1.0 <= s.min() <= s.max() <= 1.0


def test_query_similar_excludes_same_player():
    emb = _fake_embeddings()
    out = query_similar(emb, "P0", "2000-01", top_k=5)
    assert len(out) == 5
    assert not (out["player_name"] == "P0").any()


def test_write_comparisons_excludes_same_player(tmp_path):
    emb = _fake_embeddings()
    out_path = tmp_path / "cmp.csv"
    df = write_comparisons(emb, out_path)
    assert len(df) > 0
    assert not (df["query_player"] == df["match_player"]).any()
    assert df["match_era"].nunique() >= 1
