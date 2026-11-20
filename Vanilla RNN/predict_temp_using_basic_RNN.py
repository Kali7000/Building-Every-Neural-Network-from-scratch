# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:48:59 2026
@author: Vishal Wagh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_split_data(csv_file, target_col, window_len=10, train_ratio=0.8):
    """Load CSV, split chronologically, normalize without leakage, and build sliding windows."""
    df = pd.read_csv(csv_file)
    temps = df[target_col].apply(pd.to_numeric, errors='coerce').dropna().values

    split_idx = int(len(temps) * train_ratio)
    train_raw, test_raw = temps[:split_idx], temps[split_idx:]

    # Fit min/max strictly on training data
    min_val, max_val = np.min(train_raw), np.max(train_raw)
    train_norm = (train_raw - min_val) / (max_val - min_val)
    test_norm = (test_raw - min_val) / (max_val - min_val)

    x_train, y_train = create_sequences(train_norm, window_len)
    x_test, y_test = create_sequences(test_norm, window_len)

    return x_train, y_train, x_test, y_test, min_val, max_val


def create_sequences(data, window_len):
    """Slice a 1D time-series array into (X, y) sliding windows."""
    features, targets = [], []
    for i in range(len(data) - window_len):
        features.append(data[i : i + window_len])
        targets.append(data[i + window_len])
    return np.array(features), np.array(targets)


class VanillaRNN:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights small to prevent exploding gradients
        self.w_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.w_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.w_hy = np.random.randn(output_size, hidden_size) * 0.01

        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, input_window):
        """Unroll the RNN across all timesteps and return the final prediction and hidden states."""
        h = np.zeros((self.hidden_size, 1))
        hidden_states = {-1: h}

        for t in range(len(input_window)):
            x_t = np.array([[input_window[t]]])  # Shape: (1, 1)
            h = np.tanh(np.dot(self.w_xh, x_t) + np.dot(self.w_hh, h) + self.b_h)
            hidden_states[t] = h

        y_pred = np.dot(self.w_hy, h) + self.b_y
        return y_pred, hidden_states

    def bptt(self, input_window, real_target, pred_target, hidden_states):
        """Backpropagation Through Time with gradient clipping."""
        dw_xh = np.zeros_like(self.w_xh)
        dw_hh = np.zeros_like(self.w_hh)
        dw_hy = np.zeros_like(self.w_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # Output layer gradients (MSE loss: 0.5 * (y_pred - y_true)^2)
        dy = pred_target - real_target
        last_t = len(input_window) - 1
        dw_hy = np.dot(dy, hidden_states[last_t].T)
        db_y = dy

        # Pass error into the final hidden state
        dh = np.dot(self.w_hy.T, dy)

        # Walk backward through time
        for t in reversed(range(len(input_window))):
            dh_raw = (1 - hidden_states[t] ** 2) * dh
            x_t = np.array([[input_window[t]]])

            db_h += dh_raw
            dw_xh += np.dot(dh_raw, x_t.T)
            dw_hh += np.dot(dh_raw, hidden_states[t - 1].T)

            # Pass gradient backward to timestep t - 1
            dh = np.dot(self.w_hh.T, dh_raw)

        # Clip gradients to prevent exploding gradients
        for dparam in [dw_xh, dw_hh, dw_hy, db_h, db_y]:
            np.clip(dparam, -5, 5, out=dparam)

        return dw_xh, dw_hh, dw_hy, db_h, db_y

    def fit(self, x_train, y_train, lr=5e-3, epochs=30):
        """Train the RNN using Stochastic Gradient Descent."""
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(x_train)):
                y_pred, hidden_states = self.forward(x_train[i])
                total_loss += 0.5 * (y_pred[0, 0] - y_train[i]) ** 2

                dw_xh, dw_hh, dw_hy, db_h, db_y = self.bptt(
                    x_train[i], y_train[i], y_pred, hidden_states
                )

                self.w_xh -= lr * dw_xh
                self.w_hh -= lr * dw_hh
                self.w_hy -= lr * dw_hy
                self.b_h -= lr * db_h
                self.b_y -= lr * db_y

            print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {total_loss / len(x_train):.5f}")

    def predict(self, x_data):
        """Generate predictions for a set of input windows."""
        preds = []
        for i in range(len(x_data)):
            y_pred, _ = self.forward(x_data[i])
            preds.append(y_pred[0, 0])
        return np.array(preds)


def evaluate_and_plot(y_real_norm, y_pred_norm, min_val, max_val):
    """Un-normalize predictions back to Celsius, print RMSE, and plot results."""
    y_real_c = y_real_norm * (max_val - min_val) + min_val
    y_pred_c = y_pred_norm * (max_val - min_val) + min_val

    rmse = np.sqrt(np.mean((y_real_c - y_pred_c) ** 2))
    print(f"\nTest RMSE: {rmse:.2f} °C")

    plt.figure(figsize=(12, 6))
    plt.plot(y_real_c, label="Actual Temperature", color="blue", alpha=0.7)
    plt.plot(y_pred_c, label="RNN Predicted", color="red", linestyle="--", alpha=0.8)

    plt.xlabel("Days (Test Set)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Vanilla RNN: Actual vs. Predicted Temperatures (RMSE: {rmse:.2f} °C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CSV_FILE = "daily-minimum-temperatures-in-me.csv"
    TARGET_COL = "Daily minimum temperatures in Melbourne, Australia, 1981-1990"
    WINDOW_LEN = 10

    x_train, y_train, x_test, y_test, train_min_val, train_max_val = load_and_split_data(
        CSV_FILE, TARGET_COL, window_len=WINDOW_LEN
    )

    rnn = VanillaRNN(input_size=1, hidden_size=16, output_size=1)
    rnn.fit(x_train, y_train, lr=5e-3, epochs=30)

    y_pred_test = rnn.predict(x_test)
    evaluate_and_plot(y_test, y_pred_test, train_min_val, train_max_val)



# if __name__ == "__main__":
#     # 1. Generate a synthetic sine wave (1,000 points)
#     t = np.linspace(0, 100, 1000)
#     sine_wave = np.sin(t)

#     # 2. Split chronologically (80/20) and build sliding windows
#     WINDOW_LEN = 10
#     split_idx = int(len(sine_wave) * 0.8)
#     x_train, y_train = create_sequences(sine_wave[:split_idx], WINDOW_LEN)
#     x_test, y_test = create_sequences(sine_wave[split_idx:], WINDOW_LEN)

#     # 3. Train and predict
#     rnn = VanillaRNN(input_size=1, hidden_size=16, output_size=1)
#     rnn.fit(x_train, y_train, lr=5e-3, epochs=20)
#     y_pred = rnn.predict(x_test)

#     # 4. Plot
#     rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
#     print(f"\nSine Wave Test RMSE: {rmse:.4f}")

#     plt.figure(figsize=(12, 5))
#     plt.plot(y_test, label="Actual Sine Wave", color="blue", alpha=0.7)
#     plt.plot(y_pred, label="RNN Predicted", color="red", linestyle="--", alpha=0.8)
#     plt.title(f"Vanilla RNN on Sine Wave (RMSE: {rmse:.4f})")
#     plt.xlabel("Timestep (Test Set)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.show()