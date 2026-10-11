# Brain Tumor Detection CNN (From Scratch in Pure NumPy)

An end-to-end Convolutional Neural Network built completely from scratch using only Python and NumPy. This project detects and classifies brain tumors from raw MRI scans without the use of high-level deep learning frameworks like TensorFlow, PyTorch, or Keras.

## Project Overview

While modern deep learning frameworks abstract away the underlying mathematics, this project implements the entire Forward Propagation, Backpropagation, and Gradient Descent pipelines manually. 

The goal of this project was to deeply understand the Linear Algebra and Calculus that powers computer vision. By explicitly writing the nested loops for spatial convolutions and manually routing gradients through max-pooling masks, this project demonstrates a foundational mastery of neural network mechanics.

### The Dataset
The model is trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle. It classifies scans into four categories:
1. `notumor`
2. `pituitary`
3. `meningioma`
4. `glioma`

---

## Model Architecture

The network utilizes a modular 2D spatial feature extraction pipeline followed by a 1D dense classification head.

**Phase 1: Feature Extraction (2D Pipeline)**
1. **Input Layer:** `64x64` Grayscale Image (1 Channel).
2. **Conv2D (Layer 1):** 8 Kernels (Size 3x3). Extracts foundational features (edges, curves).
3. **ReLU Activation:** Element-wise non-linear clamping of negative values to `0`.
4. **Max Pooling 1:** 2x2 grid with a stride of 2. Shrinks spatial dimensions by 75% while preserving the strongest feature signals.
5. **Conv2D (Layer 2):** 16 Kernels (Size 3x3x8). Extracts complex hierarchical textures (tumor boundaries).
6. **ReLU Activation**
7. **Max Pooling 2:** 2x2 grid with a stride of 2.

**Phase 2: Classification (1D Pipeline)**
8. **Flatten:** Unrolls the 3D tensor (`16 x 14 x 14`) into a 1D vector of `3,136` input neurons.
9. **Dense (Fully Connected) Layer:** `3,136` inputs connecting to `4` output neurons.
10. **Softmax Output:** Converts raw logits into a normalized probability distribution across the 4 tumor classes.

---

## Core Concepts Implemented

This project acts as a white-box implementation of the black-box functions found in standard ML libraries. 

### 1. Spatial Convolution (Forward & Backward)
Instead of relying on optimized `im2col` matrix unrolling, the convolution is implemented using explicit sliding windows. 
*   **Forward Pass:** Calculates the element-wise sum of overlapping image patches and weight matrices.
*   **Backward Pass:** Assigns gradient "blame" by multiplying the downstream error against the original image chunks (to update weights `dW`) and the kernel weights (to route error to the previous layer `dA_prev`).

### 2. Max Pooling Gradient Routing
Max Pooling does not have weights, so it does not learn. During backpropagation, it acts as a router. The code generates a boolean mask from the forward pass to identify the single pixel that "won" the max operation, and routes 100% of the downstream error directly back to that exact coordinate, shielding the other 3 pixels from blame.

### 3. Dimensionality Collapse
Handles the transition between 4D batch tensors, 3D image volumes, and 1D Dense vectors, including the dynamic calculation of dense weights based on the output shape of the final convolutional pooling layer.

---

## How to Run the Project

Because this project uses pure Python nested `for` loops instead of C-optimized Linear Algebra libraries, it requires heavy CPU computation.

### 1. Prerequisites
You only need two libraries: NumPy for the math, and Pillow for loading the JPEG images into matrices.
```bash
pip install numpy pillow
