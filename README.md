# Plug-and-Play CSPAM Module

A PyTorch implementation of a **Channel-Spatial-Positional Attention Module (CSPAM)** for 2D feature maps.

This repository provides a lightweight attention block that refines feature representations in three stages:

1. **Channel Attention**
2. **Spatial Attention**
3. **Positional Attention**

The module is designed for convolutional neural networks and can be inserted into segmentation, classification, or detection backbones wherever feature enhancement is needed.

---

## Features

- Implemented in **PyTorch**
- Modular design with separate attention branches
- Easy to plug into existing CNN architectures
- Supports 2D feature maps of shape **(B, C, H, W)**
- Includes weight initialization function

---

## Module Overview

### 1. Channel Attention
The channel attention branch learns which channels are more informative.

- Applies global average pooling and global max pooling
- Uses a shared MLP to encode channel descriptors
- Produces a channel-wise weight map of shape `(B, C, 1, 1)`

### 2. Spatial Attention
The spatial attention branch learns where important responses are located.

- Computes average pooling and max pooling along the channel dimension
- Concatenates the pooled maps
- Uses a `7×7` convolution to produce a spatial attention map of shape `(B, 1, H, W)`

### 3. Positional Attention
The positional attention branch models long-range spatial dependencies.

- Generates three feature embeddings with `1×1` convolutions
- Builds a position-to-position attention map
- Reweights the feature representation using learned spatial relations
- Adds the refined output back through a residual connection

---

## File Structure

```text
.
├── Plug-and-Play CSPAM Module source code.py          # main implementation
└── README.md         # project documentation

