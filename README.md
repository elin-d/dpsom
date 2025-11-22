# DPSOM

A PyTorch reimplementation of **Disentangled Probabilistic Self-Organizing Map (DPSOM)**, originally proposed in TensorFlow. This implementation preserves the core methodology while providing a modern PyTorch ecosystem.

## Overview

This repository ports the original TensorFlow 1.x + TensorFlow Probability DPSOM model to native PyTorch while preserving the three‑phase workflow: autoencoder pretraining, SOM initialization, and joint optimization.​
The port maintains compatible hyperparameters and naming to simplify parity checks and reproducibility across frameworks.

## Differences vs Original TensorFlow

- **Loss Calculation**: The PyTorch model uses explicit mu/logvar and BCE‑with‑logits everywhere.
- **BatchNorm Fix**: The original TensorFlow order of BatchNorm causes a statistical mismatch between training (sparse data) and inference (dense data), which PyTorch handles strictly, breaking the model. It was reordered to the standard BatchNorm to ensure consistent feature statistics during both training and evaluation, correcting this architectural flaw.
- **Indexing**: SOM indexing in PyTorch is consistently row‑major with an optional toroidal neighbor policy.

## Hyperparameter Control

The loss function combines several objectives weighted by these parameters:

- `prior`: Reconstruction loss weight.
- `alpha`: **SOM commitment loss weight.** Controls how strongly the SOM embeddings are pulled towards the encoded representations ($z_e$).
- `beta`: KL divergence regularization (disentanglement factor).
- `gamma`: Clustering loss weight (SOM probability distribution matching).
- `theta`: Prior distribution weight.

## Improvements Over Original

- **Native PyTorch Implementation**: Clean integration with PyTorch ecosystem, automatic differentiation, and modern optimizers.
- **Stability**: Improved numerical stability in loss calculations.

## Requirements

- torch
- numpy
- scikit-learn
- sacred
- tqdm
- tensorboard

## Evaluation

The implementation includes utilities for:
- Clustering purity metrics
- Reconstruction quality assessment
- Latent space visualization
- Topology preservation evaluation

## Benchmark Results

Comparison between this PyTorch implementation and the original TensorFlow version (run with `Validation=True`).

| Dataset | Metric | Dense (Torch) | Dense (TF) | Conv (Torch) | Conv (TF) |
| :--- | :--- |:--------------| :--- | :--- | :--- |
| **MNIST** | **NMI** | **0.6946**    | 0.6919 | **0.7267** | 0.6988 |
| | **Purity** | 0.9609        | **0.9626** | **0.9833** | 0.9676 |
| **fMNIST** | **NMI** | **0.5673**    | 0.5667 | **0.5712** | 0.5667 |
| | **Purity** | 0.7738        | **0.7809** | 0.7766 | **0.7809** |

## File Structure

- `dpsom.py`: Main training script with Sacred experiment configuration
- `dpsom_model.py`: DPSOM model architecture and forward pass
- `decay_scheduler.py`: Exponential learning rate decay scheduler
- `utils.py`: Helper functions including clustering metrics

## Reference

This work is a PyTorch port of the DPSOM architecture and training procedure and references the original TensorFlow implementation for parity and validation

> Laura Manduchi, Matthias Hüser, Martin Faltys, Julia Vogt, Gunnar Rätsch,and Vincent Fortuin. 2021. T-DPSOM - An Interpretable Clustering Methodfor Unsupervised Learning of Patient Health States. InACM Conference onHealth, Inference, and Learning (ACM CHIL ’21), April 8–10, 2021, VirtualEvent, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3450439.3451872