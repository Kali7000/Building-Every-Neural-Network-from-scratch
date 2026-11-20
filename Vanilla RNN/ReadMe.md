# Vanilla Recurrent Neural Network (RNN) from Scratch

A **Many-to-One Vanilla RNN** built from scratch using only **Python and NumPy** (with Pandas for data loading and Matplotlib for visualization). 

This project implements the forward pass, **Backpropagation Through Time (BPTT)**, and **gradient clipping** without any deep learning frameworks (no PyTorch or TensorFlow) to demonstrate the underlying linear algebra and calculus of sequential neural networks.

---

## Architecture & Mathematics

The network takes a sliding window of $T$ sequential inputs (e.g., 10 days of temperatures) and predicts the next single value ($T+1$).

### 1. Forward Pass (Unrolling Through Time)
At each timestep $t \in \{0, \dots, T-1\}$, the hidden state $h_t$ combines the current input $x_t$ with the previous hidden state $h_{t-1}$:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

After processing the full sequence, the final hidden state $h_T$ is projected to a linear output:

$$\hat{y} = W_{hy} h_T + b_y$$

* **$W_{xh}$ `(16, 1)`**: Input-to-hidden weight matrix
* **$W_{hh}$ `(16, 16)`**: All-to-all recurrent feedback weight matrix
* **$W_{hy}$ `(1, 16)`**: Hidden-to-output weight matrix

### 2. Backpropagation Through Time (BPTT)
Using Mean Squared Error loss $L = \frac{1}{2}(\hat{y} - y)^2$, gradients are accumulated by walking backward from $t = T-1$ down to $t = 0$:

* Output error enters the final hidden state via $dh = W_{hy}^T \cdot dy$.
* At each step, the error passes through the $\tanh$ derivative: $dh_{\text{raw}} = (1 - h_t^2) \odot dh$.
* Gradients for $W_{xh}$, $W_{hh}$, and $b_h$ are summed across all timesteps, and the error is propagated backward in time via $dh = W_{hh}^T \cdot dh_{\text{raw}}$.
* **Gradient Clipping (`[-5, 5]`):** Applied in-place via `np.clip` to prevent exploding gradients.

---

## Key Implementation Details

* **Zero Data Leakage:** The time-series is split chronologically (`80% train / 20% test`) *before* min-max normalization. Scaling parameters are fit strictly on the training set.
* **Safe Initial State Indexing:** Hidden states are cached in a dictionary initialized with `{-1: np.zeros((hidden_size, 1))}` so that `hidden_states[t - 1]` safely resolves at `t = 0` without Python's negative list-indexing bug.
* **Modular Design:** `VanillaRNN` is decoupled from CSV loading so it can be trained on any 1D sequential dataset (e.g., real-world weather data or synthetic waveforms).

---

## Dataset

* **Daily Minimum Temperatures in Melbourne (1981–1990):** ~3,650 daily temperature observations.
* **Task:** Predict Day 11's minimum temperature using the previous 10 days (`window_len = 10`).

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib
