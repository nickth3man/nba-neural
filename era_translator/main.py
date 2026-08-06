"""CLI entry point for the Era Translator pipeline."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import evaluate
from .config import (
    DEFAULT_DB,
    DEFAULT_OUT,
    ERA_ORDER,
    MAX_EPOCHS,
    SEED,
)
from .data import load_player_seasons
from .features import (
    engineer_features,
    feature_columns,
    fit_scaler,
    make_splits,
    transform_features,
)
from .model import EraTranslatorDANN
from .train import train_model


def run(db_path, out_dir, max_epochs=MAX_EPOCHS, device="cpu"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("[1/6] Extracting player-season data (read-only DuckDB)...")
    raw = load_player_seasons(db_path)
    print(f"      {len(raw):,} player-seasons, {raw.season_year.min()} .. {raw.season_year.max()}")

    print("[2/6] Engineering features...")
    df = engineer_features(raw)
    cols = feature_columns(df)
    era_codes = df["era"].map({e: i for i, e in enumerate(ERA_ORDER)}).astype(int).values

    print("[3/6] Splitting (stratified by era)...")
    idx_train, idx_val, idx_test = make_splits(df)
    scaler = fit_scaler(df, idx_train, cols)
    X = transform_features(df, scaler, cols)
    y = df["value_target"].values.astype(np.float32)
    era = era_codes

    X_train, y_train, era_train = X[idx_train], y[idx_train], era[idx_train]
    X_val, y_val, era_val = X[idx_val], y[idx_val], era[idx_val]
    X_test, y_test, era_test = X[idx_test], y[idx_test], era[idx_test]
    print(f"      train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    print("[4/6] Training DANN...")
    model = EraTranslatorDANN(input_dim=len(cols))
    history, best_state = train_model(
        model, X_train, y_train, era_train, X_val, y_val, era_val, device=device
    )
    model_path = out_dir / "model.pt"
    torch.save(
        {"state_dict": best_state, "input_dim": len(cols), "num_eras": len(ERA_ORDER)},
        model_path,
    )
    print(f"      saved {model_path}")

    print("[5/6] Evaluating on test set...")
    model.load_state_dict(best_state)
    model.to(device)
    mse, r2, era_acc, era_bal_acc, per_decade = evaluate.evaluate_model(
        model, X_test, y_test, era_test, device=device
    )
    print(f"      test value MSE {mse:.4f} | R2 {r2:.4f}")
    print(
        f"      test era accuracy {era_acc:.4f} | balanced {era_bal_acc:.4f} "
        f"(random = {1/len(ERA_ORDER):.4f})"
    )
    for dec, acc in per_decade.items():
        print(f"        {dec}: {acc:.4f}")

    print("[6/6] Writing artifacts...")
    emb = evaluate.export_embeddings(
        df, scaler, model, X, out_dir / "embeddings.csv", device=device
    )
    print(f"      embeddings: {len(emb):,} rows -> {out_dir / 'embeddings.csv'}")
    era_probe = evaluate.era_probe_balanced(emb)
    print(f"      fresh LR era probe (balanced): {era_probe:.4f} (random = {1/len(ERA_ORDER):.4f})")
    evaluate.plot_tsne(emb, out_dir / "tsne_era.png")
    print(f"      tsne plot -> {out_dir / 'tsne_era.png'}")
    cmp = evaluate.write_comparisons(emb, out_dir / "comparisons.csv")
    print(f"      comparisons: {len(cmp):,} rows -> {out_dir / 'comparisons.csv'}")

    metrics = {
        "rows": len(df),
        "test_value_mse": mse,
        "test_value_r2": r2,
        "test_era_accuracy": era_acc,
        "test_era_balanced_accuracy": era_bal_acc,
        "fresh_lr_era_probe_balanced": era_probe,
        "random_era_accuracy": 1 / len(ERA_ORDER),
        "per_decade_era_accuracy": per_decade,
        "epochs_trained": len(history["val_loss"]),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      metrics -> {out_dir / 'metrics.json'}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Era Translator: DANN cross-era normalization")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB))
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)
    run(args.db, args.out, max_epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    sys.exit(main())
