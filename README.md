# DPSOM

A PyTorch reimplementation of **Disentangled Probabilistic Self-Organizing Map (DPSOM)**, originally proposed in TensorFlow. This implementation preserves the core methodology while providing a modern PyTorch ecosystem.

## Overview

This repository ports the original TensorFlow 1.x + TensorFlow Probability DPSOM model to native PyTorch while preserving the three‑phase workflow: autoencoder pretraining, SOM initialization, and joint optimization.​
The port maintains compatible hyperparameters and naming to simplify parity checks and reproducibility across frameworks.

## Differences vs original TensorFlow

- The PyTorch model uses explicit mu/logvar and BCE‑with‑logits everywhere.
- SOM indexing in PyTorch is consistently row‑major with an optional toroidal neighbor policy.

## Hyperparameter Control

  - `alpha`: Reconstruction loss weight
  - `beta`: KL divergence regularization (disentanglement)
  - `gamma`: Clustering loss weight
  - `theta`: Prior distribution weight

## Improvements Over Original

- **Native PyTorch Implementation**: Clean integration with PyTorch ecosystem, automatic differentiation, and modern optimizers


## Requirements

torch
numpy
scikit-learn
sacred
tqdm
tensorboard


## Evaluation

The implementation includes utilities for:
- Clustering purity metrics
- Reconstruction quality assessment
- Latent space visualization
- Topology preservation evaluation

## File Structure

- `dpsom.py`: Main training script with Sacred experiment configuration
- `dpsom_model.py`: DPSOM model architecture and forward pass
- `decay_scheduler.py`: Exponential learning rate decay scheduler
- `utils.py`: Helper functions including clustering metrics

## Reference

This work is a PyTorch port of the DPSOM architecture and training procedure and references the original TensorFlow implementation for parity and validation

> Laura Manduchi, Matthias Hüser, Martin Faltys, Julia Vogt, Gunnar Rätsch,and Vincent Fortuin. 2021. T-DPSOM - An Interpretable Clustering Methodfor Unsupervised Learning of Patient Health States. InACM Conference onHealth, Inference, and Learning (ACM CHIL ’21), April 8–10, 2021, VirtualEvent, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3450439.3451872