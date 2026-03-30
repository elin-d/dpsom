# DPSOM: Deep Probabilistic Self-Organizing Map

A PyTorch reimplementation of the **Deep Probabilistic Self-Organizing Map (DPSOM)**. This repository provides a native PyTorch implementation of the original model, preserving the core methodology while leveraging the modern ecosystem for improved stability and flexibility.

## Abstract

DPSOM combines the representational power of Variational Autoencoders (VAEs) with the topological clustering capabilities of Self-Organizing Maps (SOMs). This implementation replicates the original three-phase workflow:
1.  **Autoencoder Pretraining**: Learning a latent representation of the input space.
2.  **SOM Initialization**: Initializing the map topology based on latent codes.
3.  **Joint Optimization**: Finetuning the encoder, decoder, and SOM simultaneously.

## Architectural Implementation

This repository ports the original architecture to native PyTorch. Key architectural decisions and improvements include:

### 1. Numerical Stability & Loss Calculation
The PyTorch implementation utilizes explicit mean/log-variance parameterization and `BCEWithLogitsLoss` to ensure numerical stability during backpropagation. This addresses potential gradients issues found in legacy frameworks.

### 2. BatchNorm Rectification
A critical architectural flaw in the original implementation involved the ordering of Batch Normalization layers, which caused a statistical mismatch between training (sparse data batches) and inference (dense evaluation).

This implementation strictly enforces standard Batch Normalization ordering. This ensures that feature statistics remain consistent during both training and evaluation phases, resulting in more reliable convergence and inference metrics.

### 3. Topology & Indexing
SOM indexing is implemented using consistent row-major ordering. The model supports optional toroidal neighbor policies to handle edge cases in the map topology.

## Hyperparameters

The global objective function is a weighted sum of distinct loss components. The behavior of the model is controlled by the following hyperparameters:
 
- `prior`: Reconstruction loss weight.
- `alpha`: **SOM commitment loss weight.** Controls how strongly the SOM embeddings are pulled towards the encoded representations ($z_e$).
- `beta`: KL divergence regularization (disentanglement factor).
- `gamma`: Clustering loss weight (SOM probability distribution matching).
- `theta`: Prior distribution weight.

## Requirements

- torch
- numpy
- scikit-learn
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
| :--- | :--- |:--------------| :--- |:-------------| :--- |
| **MNIST** | **NMI** | 0.6899(46)    | 0.6919 | 0.7131(76)   | 0.6988 |
| | **Purity** | 0.9535(68)    | 0.9626 | 0.9651(98)   | 0.9676 |
| **fMNIST** | **NMI** | 0.5673        | 0.5667 | 0.5712       | 0.5667 |
| | **Purity** | 0.7738        | 0.7809 | 0.7766       | 0.7809 |

## File Structure

- `dpsom.py`: Main training script
- `dpsom_model.py`: DPSOM model architecture and forward pass
- `decay_scheduler.py`: Exponential learning rate decay scheduler
- `utils.py`: Helper functions including clustering metrics


## Reference

This work references the original methodology proposed by Manduchi et al.

> Laura Manduchi, Matthias Hüser, Martin Faltys, Julia Vogt, Gunnar Rätsch, and Vincent Fortuin. (2019). **Deep Probabilistic Self-Organizing Map**. arXiv preprint arXiv:1910.01590. https://arxiv.org/abs/1910.01590
