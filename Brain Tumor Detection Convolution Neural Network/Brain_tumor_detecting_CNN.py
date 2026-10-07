# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:30:06 2026

@author: Kali
"""

import os
import numpy as np
from PIL import Image



def one_hot_encode(labels, number_of_digits = 4):   
    ##one_Hot incoding the lables
    #number_of_digits = 4 #1,2,3,4
    
    label_lenght = len(labels)
    
    #create a 60000 by 10 matrix of zeros
    one_hot_label_matrix = np.zeros((label_lenght,number_of_digits))
    
    #change the 0 to 1 at the index of the label. ex: if label is 3 then that row == [0,0,0,1,0,0,0,0,0,0]
    one_hot_label_matrix[np.arange(label_lenght),labels] = 1.0
    
    return one_hot_label_matrix



def load_and_shuffle(base_file_path, pixl_size = 64, cache_file = "brain_tumor_cache.npz"):
    
    
    #check for cache file
    if os.path.exists(cache_file):
        print("cache file found!!!!!! Loadind....")
        
        cached_data = np.load(cache_file)
        
        X_shuffled = cached_data['X']
        Y_shuffled = cached_data['Y']
        
        return X_shuffled, Y_shuffled
    
    #define the labels
    classes = ["notumor", "pituitary", "meningioma", "glioma"]
    
    raw_images = []
    raw_labels = []
    
    #iter through folders
    for label_inter, class_name in enumerate(classes):
        folder_path = os.path.join(base_file_path, class_name)
        
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            
            #read the image, convert to grayscale (L), and resize to 64x64 to save computation
            img = Image.open(img_path).convert('L').resize((pixl_size,pixl_size))
            
            #conver to numpy array and normalize
            img_array = np.array(img)/255.0
            
            raw_images.append(img_array)
            raw_labels.append(label_inter)
            
    
    #conver to NumPy arrays
    X = np.array(raw_images)
    Y = np.array(raw_labels)
    
    ##Shuffle the dataset
    shuffled_indices = np.random.permutation(len(X))
    
    #apply the random sequence to images and labels
    X_shuffled = X[shuffled_indices]
    Y_shuffled = Y[shuffled_indices]
    
    
    #np.savez saves the np arrays as a compressed zip file (.npz)
    np.savez(cache_file,X = X_shuffled, Y = Y_shuffled)
    
    return X_shuffled, Y_shuffled


X_train, Y_train = load_and_shuffle(r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\Neural Networks\datasets\Brain Tumor MRI Dataset\Testing", 64)

Y_train_one_hot = one_hot_encode(Y_train,4)
#print(Y_train_one_hot[:5])



#### Build the layers

def init_kernals(num_kernals = 8, num_channels = 1, kernal_size = 3):
    # Shape: (Num_Kernels, Channels_In, Height, Width)
    
    kernals = np.random.randn(num_kernals, num_channels, kernal_size,kernal_size)*0.1
    
    #initate biases
    biases = np.zeros((num_kernals,num_channels))
    
    return kernals, biases
    

def convolution_feedforward_one_image(input_image, kernals, baises):
    num_kernals = kernals.shape[0]
    img_height, img_width = input_image.shape
    
    # A 3x3 kernel sliding over a 64x64 image outputs a 62x62 grid
    output_height = img_height - 3 + 1 #3 is the height of the kernal 
    output_width = img_width - 3 + 1   #3 is the width of the kernal 
    
    #create an empty 3d matric to hold all the output feature maps, in case of 8 kernals: (8,62,62)
    Z = np.zeros((num_kernals,output_height,output_width))
    
    #slide every kernal over the image
    for k in range(num_kernals):
        kernal = kernals[k]
        bias = baises[k]
        
        #slide the current kernal down the rows
        for h in range(output_height):
            #slide the kernal across the columns
            for w in range(output_width):
                
                #slice out that 3x3 chunk of pixles from the image
                image_slice = input_image[h:h+3, w:w+3]
                
                #take the elementwise multiplixation-  sum them, and add the bias
                Z[k,h,w] = np.sum(image_slice*kernal)+bias
    
    return Z


def cnn_2D_feedforward(X_volume, kernals, biases):
    """
    X_volume shape: (Channels_in, Height, Width)
    W shape: (Num_Kernels, Channels_in, Kernel_H, Kernel_W)
    B shape: (Num_Kernels, 1)
    """
    
    num_channels_in, channel_height, channel_wights  = X_volume.shape
    
    num_kernals, _, kernal_height,kernal_width = kernals.shape #_ is num of kernal_channels, if it's the first layer it would be 1, 
                                                                    #it would be 8 for the second conlv layer if the first layer had 8 kernals 
    
    output_height = channel_height - kernal_height + 1
    output_width = channel_wights - kernal_width + 1
    
    Z = np.zeros((num_kernals, output_height, output_width))
    
    for k in range(num_kernals):
        kernal = kernals[k]
        bias = biases[k]
        
        #slide the current kernal down the rows
        for h in range(output_height):
            
            #slide the kernal across the columns
            for w in range(output_width):
                
                #slice out that 3x3 chunk of pixles from the image if first layer
                # Slice through all channels simultaneously is it's the second conlv layer and the X_inout is 8 featured images
                image_slice = X_volume[:, h : h + kernal_height, w : w + kernal_width]
                
                #take the elementwise multiplixation-  sum them, and add the bias
                Z[k,h,w] = np.sum(image_slice*kernal) + bias
    
    return Z

def max_pool(input_volume,pool_size = 2, stride = 2):
    
    channels, height, width= input_volume.shape()
    
    #calculate the dimetionas of the new shrunken fetures
    
    output_height = height // stride
    output_width = width // stride
    
    #initate emepty block to store the pooled features
    P = np.zeros((channels,output_height,output_width))
    
    for c in range(channels):
        for h in range(output_height):
            for w in range(output_width):
                
                #find the starting pixels of the slice
                h_start = h*stride
                w_start = w*stride
                
                #slice it from the grid of the current channel
                chunk = input_volume[c, h_start : h_start + pool_size, w_start : w_start + pool_size]
                
                #take only the max value from the slice and save it
                P[c,h,w] = np.max(chunk)
                
                #We look at the 4 numbers inside the chunk. np.max() finds the highest number. 
                #We assign that single winning number to our new, empty container P at the exact current coordinate.
    
    return P



                








