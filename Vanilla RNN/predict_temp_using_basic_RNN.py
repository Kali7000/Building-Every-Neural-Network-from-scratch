# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 00:48:59 2026
@author: Vishal Wagh
"""

import numpy as np
import pandas as pd


daily_temp_data = pd.read_csv("daily-minimum-temperatures-in-me.csv")
temp_col = 'Daily minimum temperatures in Melbourne, Australia, 1981-1990'


temps = daily_temp_data[temp_col].apply(pd.to_numeric, errors='coerce')
temps = temps.dropna().values
min_temp = np.min(temps)
max_temp = np.max(temps)
temp_norm = (temps -min_temp)/ (np.max(temps) - min_temp)

## Create a sliding window of 10 rows, predict 11th
window_len = 10

features, target = [],[]

for i in range(len(temp_norm) - window_len):
    features.append(temp_norm[i:i+window_len])
    target.append(temp_norm[i+window_len])


##Conver the lists to numpy array for better processsing
features = np.array(features) # Shape: (len(feaures) , window_length(10)) --- lists of sequences
target = np.array(target) # shape: (len(target)) ---- 1D array



## Initilize wegiths and biases
input_size = 1 # 1 temperature value per timestep
hidden_size  = 16 # number of memory neurons
output_size = 1 # One predicted neuron

w_xh = np.random.randn(hidden_size, input_size)*0.01 ## Input to hidden
w_hh = np.random.randn(hidden_size, hidden_size)*0.01 ## Hidden to Hidden (feedback loop)
w_hy = np.random.randn(output_size, hidden_size)*0.01 # Hidden to Output

b_h = np.zeros((hidden_size,1)) #baises for input to hidden layer
b_y = np.zeros((output_size, 1)) # baises for hidden to output layer



## The forward pass
# take 10 days of temp data and predict the 11th day's temp
def forward(input_window):
    
    # Inatilize h{-1} as vector of zeros for the first time step since there's no memory
    h = np.zeros((hidden_size, 1))
    
    #store states for Backprop throught time
    hidden_states = {-1:h}
    
    #Loop through the sequence
    for t in range(len(input_window)):
        input_timestep = np.array([[input_window[t]]]) #Shape (1,1)
        h = np.tanh(np.dot(w_xh,input_timestep) + np.dot(w_hh,h) + b_h)  # SHape of h = (16,1)
        hidden_states[t] = h
    
    ##Make final prediction using the last hidden state
    y_pred = np.dot(w_hy,h) + b_y
    return y_pred, hidden_states
    
    

## Backpropogation through time (BPTT)
def bptt(input_window, real_target, pred_target, hidden_states):
    
    #inatilize gradient accumulators to zero
    dw_xh = np.zeros_like(w_xh)
    dw_hh = np.zeros_like(w_hh)
    dw_hy = np.zeros_like(w_hy)
    db_h = np.zeros_like(b_h)
    db_y = np.zeros_like(b_y)

    # Gradient MSE loss: L = 0.5*(real_traget-pred_target)^2
    # Error
    dy = pred_target - real_target #shape(1,1)
    
    #gradient for the output layer 
    last_timestep = len(input_window) - 1
    dw_hy = np.dot(dy,hidden_states[last_timestep].T)  
    db_y = dy #bias gradient is total error
    
    
    # Pass the error into the hidden state
    dh = np.dot(w_hy.T,dy)
    
    # Walk backwords through each time step
    for t in reversed(range(len(input_window))):
        # Derivative of tanh(z) is (1 - tanh(z)^2 —> (1 - h_t^2))
        dh_raw = (1 - hidden_states[t]**2) * dh
        
        x_t = np.array([[input_window[t]]])
        db_h += dh_raw
        dw_xh += np.dot(dh_raw, x_t.T)
        dw_hh += np.dot(dh_raw, hidden_states[t-1].T)
        
        #Pass the gradient to the previous timestep(t-1)
        dh = np.dot(w_hh.T, dh_raw)
    
    #gradient clipping to prevent exploding gradients
    for dparameter in[dw_xh,dw_hh,dw_hy,db_h,db_y]:
        np.clip(dparameter, -5,5,out=dparameter)
    
    return dw_xh,dw_hh,dw_hy,db_h,db_y



## The training loop

lr = 0.005
epochs = 30

##split the data
split = int(len(features)*0.8)
x_train, y_train = features[:split], target[:split]  
x_test, y_test = features[split:], target[split:]

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(x_train)):
        
        y_pred, hidden_states = forward(x_train[i])
        
        #track MSE
        total_loss +=0.5*(y_pred[0,0] - y_train[i])**2
        
        #compute gradients
        dw_xh,dw_hh,dw_hy,db_h,db_y = bptt(x_train[i],y_train[i], y_pred, hidden_states)
        
        #Update weights
        w_xh -= lr*dw_xh 
        w_hh -= lr*dw_hh
        w_hy -= lr*dw_hy
        b_h -=  lr*db_h
        b_y -=  lr*db_y

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(x_train):.5f}")  
      
    

import matplotlib.pyplot as plt

def test(x_test, y_test, t_min, t_max):
    y_real = []
    y_predicted = []
    
    for i in range(len(x_test)):
        y_real.append(y_test[i])
        y_pred, _ = forward(x_test[i])
        y_predicted.append(y_pred[0, 0])
    
    # Un-normalize back to Celsius
    y_real_c = np.array(y_real) * (t_max - t_min) + t_min
    y_pred_c = np.array(y_predicted) * (t_max - t_min) + t_min
    
    # Calculate Test RMSE in Celsius
    rmse = np.sqrt(np.mean((y_real_c - y_pred_c) ** 2))
    print(f"\nTest RMSE: {rmse:.2f} °C")

    # Plot
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

# Run evaluation
test(x_test, y_test, min_temp, max_temp)

        
        
                                               











