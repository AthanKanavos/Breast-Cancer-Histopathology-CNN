"""Train one of the three reconstructed CNN architectures."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.data import load_pcam
from src.models import build_model
from src.utils import save_history, save_metrics, save_model_summary, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--batch-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--block-dropout", type=float, default=0.25)
    parser.add_argument("--dense-dropout", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_dir = args.output_root / f"architecture_{args.architecture}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, validation_ds, test_ds = load_pcam(
        batch_size=args.batch_size,
        train_split=args.train_split,
        validation_split=args.validation_split,
        test_split=args.test_split,
        data_dir=args.data_dir,
        seed=args.seed,
    )

    model = build_model(
        args.architecture,
        learning_rate=args.learning_rate,
        block_dropout=args.block_dropout,
        dense_dropout=args.dense_dropout,
    )
    save_model_summary(model, output_dir / "model_summary.txt")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best_model.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "epoch_log.csv"),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(output_dir / "final_model.keras")
    save_history(history, output_dir)

    best_model = tf.keras.models.load_model(output_dir / "best_model.keras")
    values = best_model.evaluate(test_ds, return_dict=True, verbose=1)
    metrics = {name: float(value) for name, value in values.items()}
    save_metrics(metrics, output_dir / "metrics.json")

    print("\nTest metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
