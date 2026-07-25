# Self-Supervised Visual Representation Learning with MoCo v2

> A self-supervised computer vision pipeline that learns transferable image representations from unlabeled data using MoCo v2 and ResNet-50.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-Computer%20Vision-9658B2)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Training-76B900?logo=nvidia&logoColor=white)
![MoCo v2](https://img.shields.io/badge/Method-MoCo%20v2-2E8B57)
![ResNet-50](https://img.shields.io/badge/Backbone-ResNet--50-6A5ACD)

---

## Overview

This project explores how useful visual representations can be learned from large collections of **unlabeled images**.

Instead of training directly on class labels, the system uses **Momentum Contrast v2 (MoCo v2)** to bring different augmented views of the same image closer together in embedding space while pushing representations of different images apart.

The learned encoder is then evaluated on downstream image-classification tasks using frozen features, linear probing, and nearest-neighbor baselines.

The project combines:

- self-supervised pretraining;
- contrastive learning;
- a ResNet-50 encoder;
- momentum-based target-network updates;
- a large memory queue of negative examples;
- transfer evaluation across multiple visual benchmarks.

---

## Key Results

| Setting | Result |
|---|---:|
| Pretraining images | **~500K unlabeled images** |
| Pretraining duration | **200 epochs** |
| Backbone | **ResNet-50** |
| Memory queue | **65K representations** |
| Pretraining image size | **96 × 96** |
| Best reported downstream accuracy | **67.97% on MiniImageNet** |
| Evaluation datasets | **MiniImageNet, CUB-200, SUN397** |

The learned representations transferred across multiple downstream datasets without end-to-end fine-tuning of the encoder.

---

## Problem Formulation

Supervised image classification depends on large labeled datasets, but obtaining high-quality labels can be expensive.

This project instead learns an encoder:

```text
image -> visual embedding
```

For two augmented views of the same image, the objective encourages their embeddings to remain similar, while embeddings from unrelated images act as negatives.

The resulting feature space can then be reused for downstream tasks with limited labeled data.

---

## Methodology

### 1. Self-supervised pretraining

The model is pretrained without class labels.

Each source image is transformed into two independently augmented views. These views form a positive pair because they represent the same underlying image.

Other stored representations in the queue serve as negative examples.

### 2. ResNet-50 backbone

A ResNet-50 convolutional network acts as the visual encoder.

The classification head is replaced with a projection head during contrastive pretraining so the network can learn a representation space suitable for the MoCo objective.

### 3. Momentum encoder

MoCo maintains two encoders:

- a **query encoder**, updated through gradient descent;
- a **key encoder**, updated as an exponential moving average of the query encoder.

The momentum update stabilizes the representations used as contrastive targets.

### 4. Memory queue

The project uses a queue containing approximately **65,000 key representations**.

This provides many negative examples without requiring all of them to be processed in the same mini-batch.

The queue is updated continuously as new key embeddings are generated.

### 5. Contrastive objective

For each query representation, the matching key representation forms the positive pair.

The remaining keys in the queue act as negatives.

The model is trained to assign high similarity to the positive pair and lower similarity to unrelated samples.

### 6. Downstream evaluation

After pretraining, the encoder is frozen and evaluated using:

- **linear probing**;
- **k-nearest neighbors**;
- feature extraction at different evaluation resolutions;
- multiple downstream datasets.

This measures the quality of the learned representation independently of full end-to-end fine-tuning.

---

## Evaluation Datasets

### MiniImageNet

A compact image-classification benchmark used to measure general visual transfer.

### CUB-200

A fine-grained bird-classification dataset that tests whether the representation captures subtle visual differences.

### SUN397

A scene-recognition benchmark used to evaluate transfer beyond object-centered classification.

---

## Evaluation Strategy

### Linear probing

A linear classifier is trained on top of the frozen encoder features.

Only the linear classification layer is updated, which provides a clean measure of representation quality.

### k-Nearest Neighbors

A k-NN baseline evaluates how well examples cluster directly in the learned feature space.

### Multi-resolution evaluation

Representations are evaluated using different image resolutions, including **96 × 96** and **128 × 128** inputs.

The higher evaluation resolution improved downstream accuracy in several experiments.

---

## Repository Structure

```text
.
├── train_moco.py
├── moco/
│   ├── builder.py
│   └── loader.py
├── evaluate_linear.py
├── evaluate_knn.py
├── datasets/
├── checkpoints/
├── requirements.txt
└── README.md
```

> Update the filenames above to match the exact structure of your repository before publishing.

### `train_moco.py`

Main pretraining entry point responsible for:

- loading the unlabeled dataset;
- applying image augmentations;
- initializing the MoCo v2 model;
- running contrastive pretraining;
- saving encoder checkpoints.

### `moco/builder.py`

Defines the MoCo architecture, including:

- query and key encoders;
- momentum parameter updates;
- projection heads;
- memory queue management;
- contrastive logits.

### `moco/loader.py`

Contains augmentation and data-loading utilities used during self-supervised training.

### `evaluate_linear.py`

Extracts frozen encoder features and trains a linear classifier for downstream evaluation.

### `evaluate_knn.py`

Runs nearest-neighbor evaluation directly on the learned embedding space.

### `requirements.txt`

Lists the Python packages required to reproduce training and evaluation.

---

## Quick Start

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the data

Arrange the pretraining images using the directory structure expected by your data loader.

Example:

```text
data/
├── class_or_source_1/
│   ├── image_001.jpg
│   └── image_002.jpg
├── class_or_source_2/
│   ├── image_003.jpg
│   └── image_004.jpg
```

Labels are not used during self-supervised pretraining, but directory structure may still be required by the dataset loader.

### 5. Run MoCo v2 pretraining

```bash
python train_moco.py
```

Update the command with the actual arguments supported by your implementation.

Example:

```bash
python train_moco.py \
  --data_path data/pretrain \
  --epochs 200 \
  --image_size 96 \
  --queue_size 65536
```

### 6. Run linear evaluation

```bash
python evaluate_linear.py \
  --checkpoint checkpoints/moco_v2.pth \
  --dataset miniimagenet
```

### 7. Run k-NN evaluation

```bash
python evaluate_knn.py \
  --checkpoint checkpoints/moco_v2.pth \
  --dataset miniimagenet
```

---

## Expected Output

The training pipeline should produce:

- epoch-level contrastive-loss logs;
- saved encoder checkpoints;
- extracted downstream feature representations;
- linear-probe accuracy;
- k-NN accuracy;
- evaluation summaries across datasets and image sizes.

Example:

```text
Epoch 200/200
Contrastive Loss: 1.84
Checkpoint saved: checkpoints/moco_v2_epoch_200.pth

MiniImageNet Linear Probe Accuracy: 67.97%
```

---

## Limitations

- Pretraining was limited to 200 epochs, which is shorter than many large-scale self-supervised training regimes.
- Input resolution was smaller than standard ImageNet pretraining.
- The encoder was evaluated primarily through frozen features rather than full downstream fine-tuning.
- Performance depends heavily on augmentation design, temperature, queue size, and optimization settings.
- Results may not transfer equally well to domains substantially different from the pretraining images.
- Training remains computationally expensive despite avoiding labels.

---

## Reproducibility Notes

For a fair reproduction:

- preserve the same image augmentations;
- use the same queue size and momentum coefficient;
- record random seeds;
- keep pretraining and downstream datasets separate;
- freeze the encoder during linear probing;
- use the same image resolution and feature-extraction strategy;
- report both the checkpoint and evaluation configuration.

---

## Data and Checkpoints

The full pretraining dataset and large model checkpoints may not be included in this repository because of size and licensing constraints.

Where possible, the repository should provide:

- dataset preparation instructions;
- a small sample dataset;
- trained encoder checkpoints;
- evaluation scripts;
- configuration files.

---

## Disclaimer

This project was developed for educational and portfolio purposes. Reported performance reflects the specific datasets, training setup, and evaluation protocol used in this implementation.
