
# **Plug-and-Play CSPAM Module**

A PyTorch implementation of a **plug-and-play Channel-Spatial-Positional Attention Module (CSPAM)** for convolutional neural networks.

This repository provides a lightweight attention block that can be easily inserted into existing deep learning models for feature refinement. The module combines **channel attention**, **spatial attention**, and **positional attention** in a sequential manner, and is suitable for a wide range of computer vision tasks, especially medical image analysis and image segmentation.

---
![Figure](https://github.com/Edward-E-S-Wang/Plug-and-Play-CSPAM-Module/blob/main/CSPAM.png)
Figure 1: The schematic figure of CSPAM

## Overview

Attention mechanisms are widely used to enhance feature representation in deep learning models. Different attention branches focus on different aspects of the feature map:

- **Channel attention** emphasizes informative channels
- **Spatial attention** highlights important regions
- **Positional attention** models long-range dependencies across spatial positions

This project integrates these three branches into a unified module called **CSPAM**. The module is designed to be **plug-and-play**, meaning it can be inserted into existing architectures with minimal modification.

The current implementation is written in **PyTorch** and is suitable for:

- medical image segmentation
- image classification
- object detection
- feature refinement in encoder-decoder networks
- other dense prediction tasks

---

## Module Structure

The processing flow of CSPAM is:

```text
Input Feature
   ↓
Channel Attention
   ↓
Spatial Attention
   ↓
Positional Attention
   ↓
Output Feature
````

The three branches are applied sequentially so that channel-wise, region-wise, and position-wise information can be progressively enhanced.

---

## Main Components

This implementation contains the following modules:

* `ChannelAttention`
* `SpatialAttention`
* `PositionalAttention`

---

## Method Summary

### 1. Channel Attention

The channel attention branch uses both **global average pooling** and **global max pooling** to extract channel-wise descriptors. These two descriptors are passed through a shared multilayer perceptron (MLP), and their outputs are fused to produce a channel attention map.

**Input shape**

```text
(B, C, H, W)
```

**Output shape**

```text
(B, C, 1, 1)
```

**Main idea**

* captures global channel statistics
* uses both global average pooling and global max pooling
* generates channel-wise weights in the range `[0, 1]`

---

### 2. Spatial Attention

The spatial attention branch applies **channel-wise average pooling** and **channel-wise max pooling** to preserve complementary spatial information. The resulting two maps are concatenated and passed through a `7×7` convolution to generate a spatial attention map.

**Input shape**

```text
(B, C, H, W)
```

**Output shape**

```text
(B, 1, H, W)
```

**Main idea**

* highlights important spatial regions
* keeps complementary spatial cues
* produces a spatial weight map in the range `[0, 1]`

---

### 3. Positional Attention

The positional attention branch models long-range dependencies between spatial positions. Three `1×1` convolutions are used to produce feature embeddings, and an attention matrix is constructed to describe interactions between all positions. The resulting attention map is used to reweight features, followed by a residual connection.

**Input shape**

```text
(B, C, H, W)
```

**Output shape**

```text
(B, C, H, W)
```

**Main idea**

* captures global contextual dependencies
* models position-to-position relationships
* enhances feature representation with residual learning

---

## Repository Structure

```text
Plug-and-Play-CSPAM-Module/
├─ Plug-and-Play CSPAM Module source code.py
└─ README.md
```

---

## Requirements

* Python 3.8 or later
* PyTorch 1.8 or later

This implementation uses only PyTorch and standard Python libraries.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Plug-and-Play-CSPAM-Module.git
cd Plug-and-Play-CSPAM-Module
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

```txt
torch>=1.8.0
```

---

## Usage

### Basic Example

```python
import torch
from cspam import CSPAM

x = torch.randn(1, 512, 28, 28)

model = CSPAM(in_channels=512, reduction=2)
model.init_weights()

out = model(x)

print("Input shape :", x.shape)
print("Output shape:", out.shape)
```

### Example Output

```text
Input shape : torch.Size([1, 512, 28, 28])
Output shape: torch.Size([1, 512, 28, 28])
```

---

## How to Integrate into Your Network

CSPAM is designed as a plug-and-play module, so it can be inserted into existing models with minimal changes. A common strategy is to place it after a convolution block, feature fusion block, encoder stage, or decoder stage.

### Example

```python
import torch.nn as nn
from cspam import CSPAM

class SimpleBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = CSPAM(in_channels=in_channels, reduction=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        return x
```

### Typical insertion positions

CSPAM can be placed in:

* encoder blocks
* decoder blocks
* bottleneck layers
* skip-connection fusion blocks
* backbone feature refinement stages

---

## Initialization

The `init_weights()` method is provided to initialize module parameters.

### Initialization rules

* **Conv2d**: Kaiming normal initialization
* **Linear**: normal initialization with small standard deviation
* **BatchNorm2d**: weight = 1, bias = 0

### Example

```python
model = CSPAM(in_channels=512, reduction=2)
model.init_weights()
```

---

## Input and Output

### Input

The module expects a 4D tensor:

```text
(B, C, H, W)
```

where:

* `B` = batch size
* `C` = number of channels
* `H` = height
* `W` = width

### Output

The output tensor has the same shape as the input:

```text
(B, C, H, W)
```

This makes CSPAM easy to insert into existing models without changing downstream tensor dimensions.

---

## Notes

* `in_channels` must be divisible by `reduction`
* the current spatial attention branch uses a fixed `7×7` convolution
* the positional attention branch builds an attention matrix of size `N × N`, where `N = H × W`
* for large feature maps, positional attention may increase memory consumption
* if memory is limited, it is recommended to use CSPAM on lower-resolution feature maps or bottleneck features

---

## Recommended Use Cases

This module is especially suitable for tasks where feature refinement is important, such as:

* lesion segmentation
* tumor segmentation
* organ segmentation
* medical image classification
* visual feature enhancement in CNNs
* general dense prediction problems

Because the module is independent and self-contained, it can also be adapted to other vision architectures beyond medical imaging.

---

## Running the Included Test Example

If your `cspam.py` file includes the following test block:

```python
if __name__ == "__main__":
    x = torch.randn(1, 512, 28, 28)
    model = CSPAM(in_channels=512, reduction=2)
    model.init_weights()

    out = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", out.shape)
```

you can run it directly from the command line:

```bash
python cspam.py
```

This is a simple way to verify that the module works correctly.

---

## Possible Future Improvements

Possible extensions for future versions include:

* support for configurable spatial attention kernel sizes
* lighter positional attention for large feature maps
* optional normalization layers inside attention branches
* direct integration examples for U-Net, ResNet, and other backbones
* benchmark experiments on public datasets

---
## Citation
If you use this code, this implementation strategy, or a modified version of it in academic work, please cite the original article:


Wang, Y., Wen, Z., Bao, S. et al. *Diffusion-CSPAM U-Net: A U-Net model integrated hybrid attention mechanism and diffusion model for segmentation of computed tomography images of brain metastases*. Radiation Oncology 20, 50 (2025). DOI: 10.1186/s13014-025-02622-x
