"""Utility functions for reproducibility and experiment output."""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_model_summary(model: tf.keras.Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "history.csv", index=False)

    for metric in ("loss", "accuracy", "auc", "precision", "recall"):
        validation_metric = f"val_{metric}"
        if metric not in history_df or validation_metric not in history_df:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(history_df[metric], label=f"training {metric}")
        plt.plot(history_df[validation_metric], label=f"validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"Training and validation {metric}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_curve.png", dpi=160)
        plt.close()


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
