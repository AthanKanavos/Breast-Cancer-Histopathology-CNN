"""Evaluate a saved Keras model on the PCam test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.data import load_pcam


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, _, test_ds = load_pcam(
        batch_size=args.batch_size,
        train_split="train[:1%]",
        validation_split="validation[:1%]",
        test_split=args.test_split,
        data_dir=args.data_dir,
    )

    model = tf.keras.models.load_model(args.model_path)
    values = model.evaluate(test_ds, return_dict=True, verbose=1)
    metrics = {name: float(value) for name, value in values.items()}

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
