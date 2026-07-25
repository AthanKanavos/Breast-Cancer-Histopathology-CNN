# Breast Cancer Classification of Histopathological Images

A **reconstructed implementation** of the three CNN architectures described in:

> *Breast Cancer Classification of Histopathological Images using Deep Convolutional Neural Networks*

The repository trains and evaluates three TensorFlow/Keras models on the
PatchCamelyon (PCam) dataset.

## Important reproducibility note

The paper specifies the high-level layer patterns, the dataset, batch sizes,
epochs, and evaluation metrics, but it does **not** fully specify every
implementation detail required to recover the original source code exactly
(for example: number of filters, kernel sizes, activations, dropout rates,
optimizer, learning rate, initialization, preprocessing, and random seed).

Therefore, this repository is a **faithful reconstruction**, not a claim that
it is the deleted original code. All assumptions are clearly documented below.

## Architectures

### Architecture 1

```text
(Conv2D x2 -> BatchNormalization -> MaxPooling2D -> Dropout) x2
-> GlobalAveragePooling2D -> Flatten -> Dense(256) -> Dropout -> Output
```

### Architecture 2

```text
((Conv2D -> BatchNormalization) x2 -> MaxPooling2D -> Dropout) x2
-> GlobalAveragePooling2D -> Flatten -> Dense(256) -> Dropout -> Output
```

### Architecture 3

```text
(Conv2D x3 -> BatchNormalization -> MaxPooling2D -> Dropout) x2
-> Conv2D x2 -> BatchNormalization -> MaxPooling2D -> Dropout
-> GlobalAveragePooling2D -> Flatten -> Dense(256) -> Dropout -> Output
```

## Reconstruction assumptions

- Input shape: `96 x 96 x 3`
- Binary classification with one sigmoid output
- Convolution filters: `32`, `64`, and `128`
- Kernel size: `3 x 3`
- Padding: `same`
- Activation: ReLU
- Pooling: `2 x 2`
- Convolutional-block dropout: `0.25`
- Dense-layer dropout: `0.50`
- Optimizer: Adam
- Learning rate: `0.001`
- Loss: binary cross-entropy
- Pixel scaling: `[0, 255] -> [0, 1]`
- Default random seed: `42`

These values can be changed from the command line.

## Dataset

TensorFlow Datasets downloads PCam automatically. The full dataset is large
(approximately 7–8 GB), so make sure you have enough disk space.

Expected splits:

| Split | Images |
|---|---:|
| Train | 262,144 |
| Validation | 32,768 |
| Test | 32,768 |

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick smoke test

This uses only small portions of the dataset:

```bash
python train.py --architecture 1 --epochs 1 \
  --train-split "train[:1%]" \
  --validation-split "validation[:10%]"
```

## Full training

Architecture 1, batch size 128, 20 epochs:

```bash
python train.py --architecture 1 --batch-size 128 --epochs 20
```

Architecture 2, batch size 256:

```bash
python train.py --architecture 2 --batch-size 256 --epochs 20
```

Architecture 3, batch size 128:

```bash
python train.py --architecture 3 --batch-size 128 --epochs 20
```

## Evaluation

```bash
python evaluate.py \
  --model-path outputs/architecture_1/best_model.keras \
  --batch-size 128
```

## Outputs

Training creates:

```text
outputs/architecture_N/
├── best_model.keras
├── final_model.keras
├── history.csv
├── metrics.json
├── model_summary.txt
└── training_curves.png
```

## Project structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── evaluate.py
└── src/
    ├── __init__.py
    ├── data.py
    ├── models.py
    └── utils.py
```

## Citation

@inproceedings{DBLP:conf/seeda/KanavosKPM22,
  author       = {Athanasios Kanavos and
                  Efstratios Kolovos and
                  Orestis Papadimitriou and
                  Manolis Maragoudakis},
  title        = {Breast Cancer Classification of Histopathological Images using Deep
                  Convolutional Neural Networks},
  booktitle    = {7th South-East Europe Design Automation, Computer Engineering, Computer
                  Networks and Social Media Conference, {SEEDA-CECNSM} 2022, Ioannina,
                  Greece, September 23-25, 2022},
  pages        = {1--6},
  publisher    = {{IEEE}},
  year         = {2022},
  url          = {https://doi.org/10.1109/SEEDA-CECNSM57760.2022.9932898},
  doi          = {10.1109/SEEDA-CECNSM57760.2022.9932898},
  timestamp    = {Mon, 03 Mar 2025 21:21:16 +0100},
  biburl       = {https://dblp.org/rec/conf/seeda/KanavosKPM22.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

## License

MIT License. This license applies to this reconstructed implementation.
Before publishing, confirm that all co-authors agree with making the
implementation public and that no restricted data or third-party code is
included.
