# Handwritten Digit Recognition: Neural Network from Scratch

A fully connected neural network that classifies handwritten digits from the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset, built using **only NumPy**. There is no TensorFlow, PyTorch, or scikit-learn: forward propagation, the loss function, backpropagation, and gradient descent are all implemented by hand.

The goal of this project is to understand what deep learning frameworks do under the hood.

---

## Table of Contents

- [Features](#features)
- [Network Architecture](#network-architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [Evaluation](#evaluation)
- [Results](#results)
- [Key Lessons](#key-lessons)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Features

- Neural network written in pure NumPy
- One-hot label encoding
- ReLU hidden layer and softmax output layer
- Categorical cross-entropy loss
- Manual backpropagation using the chain rule
- Full-batch gradient descent
- Detailed test evaluation: confusion matrix, per-class precision/recall/F1, top confusions, and confidence analysis

---

## Network Architecture

```
Input (784)  ──►  Hidden (128, ReLU)  ──►  Output (10, Softmax)
 28×28 pixels        W1: 784×128              W2: 128×10
 flattened           B1: 1×128                B2: 1×10
```

| Layer  | Shape        | Activation | Parameters |
|--------|--------------|------------|-----------:|
| Input  | 784          | none       | 0          |
| Hidden | 128          | ReLU       | 100,480    |
| Output | 10           | Softmax    | 1,290      |
| **Total** |           |            | **101,770** |

---

## How It Works

The code is organized into six phases.

### Phase 1: Data Preparation
- Load the MNIST IDX files with `idx2numpy`.
- Flatten each 28×28 image into a 784-length vector.
- Normalize pixel values from `[0, 255]` to `[0, 1]`.
- One-hot encode the labels (e.g. `3 → [0,0,0,1,0,0,0,0,0,0]`).

### Phase 2: Parameter Initialization
Weights are drawn from a small random normal distribution, and biases start at zero.

### Phase 3: Forward Propagation

```
Z1 = X · W1 + B1
A1 = ReLU(Z1)
Z2 = A1 · W2 + B2
A2 = softmax(Z2)
```

Softmax is computed in a numerically stable way by subtracting the row maximum before exponentiating.

### Phase 4: Loss Function
Categorical cross-entropy, averaged over the batch:

```
L = -(1/m) · Σ Y · log(A2)
```

### Phase 5: Backpropagation

```
dZ2 = (A2 - Y) / m
dW2 = A1ᵀ · dZ2
dB2 = Σ dZ2
dZ1 = (dZ2 · W2ᵀ) ⊙ ReLU'(Z1)
dW1 = Xᵀ · dZ1
dB1 = Σ dZ1
```

The gradient of softmax combined with cross-entropy simplifies neatly to `A2 - Y`.

### Phase 6: Gradient Descent
Each epoch runs a forward pass, computes gradients, and updates every parameter:

```
W ← W - learning_rate · dW
B ← B - learning_rate · dB
```

---

## Project Structure

```
.
├── mnist_nn.py                  # Full implementation (training + evaluation)
├── train-images.idx3-ubyte      # MNIST training images (60,000)
├── train-labels.idx1-ubyte      # MNIST training labels
├── t10k-images.idx3-ubyte       # MNIST test images (10,000)
├── t10k-labels.idx1-ubyte       # MNIST test labels
└── README.md
```

> Rename `mnist_nn.py` to match your actual script name.

---

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- idx2numpy

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

pip install numpy matplotlib idx2numpy
```

### Download the Dataset

Download the four MNIST files and place them (uncompressed) in the project root:

| File | Contents |
|------|----------|
| `train-images.idx3-ubyte` | 60,000 training images |
| `train-labels.idx1-ubyte` | 60,000 training labels |
| `t10k-images.idx3-ubyte`  | 10,000 test images |
| `t10k-labels.idx1-ubyte`  | 10,000 test labels |

The files are mirrored in many places online. Search for "MNIST IDX files" or use the original source at <http://yann.lecun.com/exdb/mnist/>. If your files end in `.gz`, extract them first.

---

## Usage

```bash
python mnist_nn.py
```

This will:

1. Load and preview the first five training images.
2. Train the network and print loss and accuracy every 100 epochs.
3. Evaluate on the 10,000 test images and print the full report.

Example training output:

```
Epoch   0 | Loss:   2.3012 | Accuracy: 10.52%
Epoch 100 | Loss:   0.xxxx | Accuracy: xx.xx%
...
```

---

## Hyperparameters

| Parameter        | Value  | Notes |
|------------------|--------|-------|
| Hidden units     | 128    | Defined in `initate_paramaters()` |
| Learning rate    | 0.1    | Full-batch gradient descent |
| Epochs           | 1000   | One epoch = one pass over all 60,000 images |
| Weight init      | `randn × 0.1` | He initialization (`√(2/n_in)`) is a good alternative for ReLU |
| Batch size       | 60,000 | Full batch |

---

## Evaluation

`test_model()` prints a full report, not just a single accuracy number:

- **Overall metrics:** accuracy, mean cross-entropy loss, comparison to random-guess and majority-class baselines, NaN check
- **Confusion matrix:** rows are the true digit and columns are the predicted digit
- **Per-class precision, recall, and F1:** reveals which digits the model struggles with
- **Top 5 confusions:** the most frequent mistakes (e.g. 4 mistaken for 9)
- **Confidence analysis:** average confidence when right vs. wrong, and the most confidently wrong predictions

---

## Results

> Fill these in after your final run.

| Metric                  | Value |
|-------------------------|-------|
| Training accuracy       | `__%` |
| Test accuracy           | `__%` |
| Test cross-entropy loss | `__`  |
| Weakest digit (recall)  | `__`  |

A simple one-hidden-layer network like this typically reaches roughly **92-97% test accuracy** on MNIST.

---

## Key Lessons

Building this from scratch surfaced several practical issues that frameworks usually hide:

1. **Normalize your inputs.** Raw pixel values (0-255) cause huge pre-activations, which makes `exp()` overflow and produces `NaN`.
2. **Use a numerically stable softmax.** Subtracting the row maximum gives an identical result without overflow.
3. **Average the gradients over the batch.** Summing over 60,000 samples makes the effective learning rate enormous. Averaging makes the learning rate independent of dataset size.
4. **Watch for `NaN` and a loss stuck near ln(10) ≈ 2.30.** That value is what random guessing over 10 classes produces, so it signals the model isn't learning.
5. **Accuracy alone isn't enough.** The confusion matrix and confidence analysis show *how* the model fails.

---

## Future Improvements

- [ ] Mini-batch / stochastic gradient descent
- [ ] Train / validation split and early stopping
- [ ] He weight initialization
- [ ] Momentum or Adam optimizer
- [ ] Learning-rate decay
- [ ] L2 regularization or dropout
- [ ] Additional hidden layers
- [ ] Plot loss and accuracy curves
- [ ] Save and load trained weights (`np.save` / `np.load`)
- [ ] Visualize misclassified digits
- [ ] Data augmentation (small shifts and rotations)

---

## Acknowledgements

- [MNIST database](http://yann.lecun.com/exdb/mnist/) by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- [`idx2numpy`](https://pypi.org/project/idx2numpy/) for reading the IDX file format

---
