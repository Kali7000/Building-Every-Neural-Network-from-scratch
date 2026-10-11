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



### Build the layers


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
    kernals shape: (Num_Kernels, Channels_in, Kernel_H, Kernel_W)
    biases shape: (Num_Kernels, 1)
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


def init_kernals(num_kernals, num_channels, kernal_size = 3):
    # Shape: (Num_Kernels, Channels_In, Height, Width)  
    kernals = np.random.randn(num_kernals, num_channels, kernal_size,kernal_size)*0.1
    #initate biases
    biases = np.zeros((num_kernals,1))
    return kernals, biases


def init_dense_weights(num_flattened_inputs, num_classes=4):
    #Dense Fully Connected Layer weights and biases
    # W3 Shape: (Inputs, Outputs)
    # Example: (3136 rows, 4 columns)
    W = np.random.randn(num_flattened_inputs,num_classes)*0.1
    B = np.zeros((1,num_classes))
    return W,B



def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

 


       
### Backpropogation

def backprop_max_pooling(dP,A_prev,pool_size = 2, stride = 2):    
    """
    dP: The incoming error from the layer ahead (e.g., shape 16x14x14)
    A_prev: The original activations BEFORE pooling (e.g., shape 16x29x29)
    """
    num_channels, output_height, output_width = dP.shape
    
    #create an empty container for the backward error matching the shape of A_prev that went into the the pooling function
    dA_prev = np.zeros_like(A_prev)
    
    for c in range(num_channels):
        for h in range(output_height):
            for w in range(output_width):
                h_start = h * stride
                w_start = w * stride
                
                #1. get the original chunk back using the A_prev
                chunk = A_prev[c, h_start : h_start + pool_size, w_start : w_start + pool_size]
                
                #2. find the highest number in the chunk
                max_val = np.max(chunk)
                
                #3. create a Mask where max is true the others are false
                mask = (chunk==max_val)
                
                #4. route the backpropogating error only to the max pixel, turn others to 0
                dA_prev[c,h_start:h_start+pool_size,w_start:w_start+pool_size] += mask *dP[c,h,w]
                
                
    return dA_prev


def relu_derivative(z):
    #Returns 1 if Z > 0, otherwise returns 0.
    return (z>0).astype(float)  ## .astype(float) turns True into 1.0 and False into 0.0




def backprop_convolution(dZ,A_prev,kernals):
    """
    dZ: The incoming error (e.g., 16 feature maps of error)
    A_prev: The input volume to this layer from the forward pass
    W: The kernels/weights of this layer
    """
    
    dW = np.zeros_like(kernals)
    db = np.zeros((kernals.shape[0],1))
    dA_prev = np.zeros_like(A_prev)
    
    






def gradient_decent(X,Y,Y_label,epochs):
  
    kernals1, biases1 = init_kernals(8,1) #8 kernals; 1 channel
    # Layer 1 Input: You feed it Grayscale MRIs, which have 1 color channel.
    # Layer 1 Processing: You initiate 8 kernels (num_kernals = 8).
    # Layer 1 Output: Because you used 8 kernels, the image is transformed into 8 stacked feature maps. The output now has 8 channels.
    # Layer 2 Input: Since the data arriving at Layer 2 has 8 channels, the Layer 2 kernels must be 8 channels deep to process it.
    kernals2, biases2 = init_kernals(16,8) #16 kernals to find complex features ; 8 channels = 8 feature output of layer 1
    
    dense_weights, dense_biases = init_dense_weights(3136)
    
    for i in range(epochs):   
        #Layer 1
        Z1 = cnn_2D_feedforward(X, kernals1, biases1) 
            #Relu activation
        A1 = np.maximum(0,Z1)      
            #Pooling Layer
        P1 = max_pool(A1)
        
        #Layer 2
        Z2 = cnn_2D_feedforward(P1, kernals2, biases2)
        A2 = np.maximum(0,Z2)
        P2 = max_pool(A2)
 
        #Layer 3: The dense layer
        # Bridge: Flatten the 3D block into a 1D array
        # P2 shape is (16, 14, 14). 16 * 14 * 14 = 3,136 pixels
        flattend_p2 = P2.flatten().reshape(1,-1)  # Shape becomes (1, 3136)     
        Z3 = np.dot(flattend_p2,dense_weights)+dense_biases
        A3 = softmax(Z3)
        
        
        
        #Output error
        dZ3 = A3 - Y
        
        #Dense layer gradient/Blame
        dW3 = np.dot(flattend_p2.T,dZ3)
        dB3 = np.sum(dZ3,axis=0,keepdims=True)
        
        #Push error backwords to the flatten layer
        dFlat = np.dot(dZ3,dense_weights.T)
        
        ## Unflatten, reshape the flattend_p2 back to it's 3D metrix shape
        dP2 = dFlat.reshape(16, 14, 14)
    
        
    

        
        
        
        



base_file_path = r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\Neural Networks\datasets\Brain Tumor MRI Dataset\Testing"
resize_images_to = 64        #64x64 pixels

X_train, Y_train = load_and_shuffle(base_file_path, resize_images_to)

Y_train_one_hot = one_hot_encode(Y_train,4)
#print(Y_train_one_hot[:5])



epochs = 100




































