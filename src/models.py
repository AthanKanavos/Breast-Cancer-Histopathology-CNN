"""CNN model definitions reconstructed from the architecture table in the paper."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def _conv(
    x: tf.Tensor,
    filters: int,
    *,
    batch_norm: bool,
    name: str,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size=(3, 3),
        padding="same",
        use_bias=not batch_norm,
        kernel_initializer="he_normal",
        name=f"{name}_conv",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def _classification_head(
    x: tf.Tensor,
    *,
    dense_units: int,
    dense_dropout: float,
) -> tf.Tensor:
    # The paper explicitly lists both GlobalAveragePooling2D and Flatten.
    # Flatten is retained for structural fidelity, although it is effectively
    # a no-op after global average pooling.
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_256")(x)
    x = layers.Dropout(dense_dropout, name="dense_dropout")(x)
    return layers.Dense(1, activation="sigmoid", name="prediction")(x)


def build_architecture_1(
    input_shape: tuple[int, int, int] = (96, 96, 3),
    filters: tuple[int, int, int] = (32, 64, 128),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
    dense_units: int = 256,
) -> tf.keras.Model:
    """(Conv2D x2 - BN - MaxPool - Dropout) x2."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block_index, num_filters in enumerate(filters[:2], start=1):
        x = _conv(x, num_filters, batch_norm=False, name=f"block{block_index}_conv1")
        x = _conv(x, num_filters, batch_norm=False, name=f"block{block_index}_conv2")
        x = layers.BatchNormalization(name=f"block{block_index}_bn")(x)
        x = layers.MaxPooling2D((2, 2), name=f"block{block_index}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block{block_index}_dropout")(x)

    outputs = _classification_head(
        x, dense_units=dense_units, dense_dropout=dense_dropout
    )
    return models.Model(inputs, outputs, name="paper_architecture_1")


def build_architecture_2(
    input_shape: tuple[int, int, int] = (96, 96, 3),
    filters: tuple[int, int, int] = (32, 64, 128),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
    dense_units: int = 256,
) -> tf.keras.Model:
    """((Conv2D - BN) x2 - MaxPool - Dropout) x2."""
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block_index, num_filters in enumerate(filters[:2], start=1):
        x = _conv(x, num_filters, batch_norm=True, name=f"block{block_index}_conv1")
        x = _conv(x, num_filters, batch_norm=True, name=f"block{block_index}_conv2")
        x = layers.MaxPooling2D((2, 2), name=f"block{block_index}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block{block_index}_dropout")(x)

    outputs = _classification_head(
        x, dense_units=dense_units, dense_dropout=dense_dropout
    )
    return models.Model(inputs, outputs, name="paper_architecture_2")


def build_architecture_3(
    input_shape: tuple[int, int, int] = (96, 96, 3),
    filters: tuple[int, int, int] = (32, 64, 128),
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
    dense_units: int = 256,
) -> tf.keras.Model:
    """
    (Conv2D x3 - BN - MaxPool - Dropout) x2
    - Conv2D x2 - BN - MaxPool - Dropout.
    """
    inputs = layers.Input(shape=input_shape, name="image")
    x = inputs

    for block_index, num_filters in enumerate(filters[:2], start=1):
        for conv_index in range(1, 4):
            x = _conv(
                x,
                num_filters,
                batch_norm=False,
                name=f"block{block_index}_conv{conv_index}",
            )
        x = layers.BatchNormalization(name=f"block{block_index}_bn")(x)
        x = layers.MaxPooling2D((2, 2), name=f"block{block_index}_pool")(x)
        x = layers.Dropout(block_dropout, name=f"block{block_index}_dropout")(x)

    for conv_index in range(1, 3):
        x = _conv(
            x,
            filters[2],
            batch_norm=False,
            name=f"block3_conv{conv_index}",
        )
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.MaxPooling2D((2, 2), name="block3_pool")(x)
    x = layers.Dropout(block_dropout, name="block3_dropout")(x)

    outputs = _classification_head(
        x, dense_units=dense_units, dense_dropout=dense_dropout
    )
    return models.Model(inputs, outputs, name="paper_architecture_3")


def build_model(
    architecture: int,
    *,
    learning_rate: float = 1e-3,
    block_dropout: float = 0.25,
    dense_dropout: float = 0.50,
) -> tf.keras.Model:
    builders = {
        1: build_architecture_1,
        2: build_architecture_2,
        3: build_architecture_3,
    }
    if architecture not in builders:
        raise ValueError("architecture must be one of: 1, 2, 3")

    model = builders[architecture](
        block_dropout=block_dropout,
        dense_dropout=dense_dropout,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
