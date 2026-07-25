"""PatchCamelyon input pipeline."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE


def _preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, (1,))
    return image, label


def load_pcam(
    *,
    batch_size: int,
    train_split: str = "train",
    validation_split: str = "validation",
    test_split: str = "test",
    data_dir: str | None = None,
    shuffle_buffer: int = 8192,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load train, validation, and test splits using TensorFlow Datasets."""
    datasets = tfds.load(
        "patch_camelyon",
        split=[train_split, validation_split, test_split],
        as_supervised=True,
        data_dir=data_dir,
        shuffle_files=True,
    )
    train_ds, validation_ds, test_ds = datasets

    train_ds = (
        train_ds
        .shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    validation_ds = (
        validation_ds
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    test_ds = (
        test_ds
        .map(_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    return train_ds, validation_ds, test_ds
