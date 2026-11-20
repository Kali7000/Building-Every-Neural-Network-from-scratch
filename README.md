# Deep Learning & Machine Learning from Scratch (Pure NumPy)

An ongoing collection of core neural network and machine learning architectures implemented **from scratch using only Python and NumPy**—without high-level frameworks like PyTorch, TensorFlow, or Keras.

Every project in this repository is a white-box implementation designed to expose the linear algebra, calculus, and optimization mechanics that modern frameworks abstract away: manual forward propagation, backpropagation, gradient routing, and numerical stability.

> **Status:** Actively maintained. New architectures and algorithms are continuously added as standalone project directories.

---

## Project Directory

| Project | Architecture | Task & Dataset | Key Concepts Implemented |
| :--- | :--- | :--- | :--- |
| **[Handwritten Digit Recognition](https://github.com/Kali7000/Building-Every-Neural-Network-from-scratch/tree/main/Handwritten%20Digit%20Detection%20Multi-layer%20perceptron)** | Fully Connected MLP (`784 → 128 → 10`) | 10-class Image Classification (MNIST) | Manual backprop, ReLU, numerically stable Softmax, Cross-Entropy loss, full-batch GD, confusion matrix & confidence analysis |
| **[Brain Tumor Detection CNN](https://github.com/Kali7000/Building-Every-Neural-Network-from-scratch/tree/main/Brain%20Tumor%20Detection%20Convolution%20Neural%20Network)** | 2D Convolutional Neural Network | 4-class MRI Classification (Kaggle Brain Tumor MRI) | Explicit sliding-window Conv2D (forward & backward), Max Pooling boolean mask gradient routing, 3D-to-1D flattening |
| **[Vanilla RNN](https://github.com/Kali7000/Building-Every-Neural-Network-from-scratch/tree/main/Vanilla%20RNN)** | Many-to-One Recurrent Neural Network | Time-Series Forecasting (Melbourne Daily Min. Temps) | Unrolling through time, all-to-all recurrent feedback ($W_{hh}$), Backpropagation Through Time (BPTT), gradient clipping, leakage-free splitting |


---

## How to Navigate This Repository

Each project is completely self-contained in its own top-level folder so you can run, study, or modify any architecture independently:

```text
.
├── mnist-nn-from-scratch/          # Fully Connected Neural Network (MLP)
│   ├── MNIST_digit_deteting_multi-layer_Perceptron.py
│   └── README.md
├── brain-tumor-cnn-from-scratch/   # 2D Convolutional Neural Network (CNN)
│   ├── Brain_tumor_detecting_CNN.py
│   └── README.md
├── vanilla-rnn-from-scratch/       # Vanilla Recurrent Neural Network (RNN)
│   ├── vanilla_rnn.py
│   └── README.md
└── README.md                       # You are here
