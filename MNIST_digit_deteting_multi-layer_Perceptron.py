# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 19:02:40 2026

@author: Kali
"""

#### Building Handwritten digit detection Neural Net from scrath


import idx2numpy
import matplotlib.pyplot as plt
import numpy as np

##Helper_functions

def one_hot_encode(labels, number_of_digits = 10):   
    ##one_Hot incoding the lables
    #number_of_digits = 10 #1,2,3,4,5,6,7,8,9,0
    
    label_lenght = len(labels)
    
    #create a 60000 by 10 matrix of zeros
    one_hot_label_matrix = np.zeros((label_lenght,number_of_digits))
    
    #change the 0 to 1 at the index of the label. ex: if label is 3 then that row == [0,0,0,1,0,0,0,0,0,0]
    one_hot_label_matrix[np.arange(label_lenght),labels] = 1.0
    
    return one_hot_label_matrix


#--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------

#### Phase 2: Network Initilization


def initate_paramaters():   
    #Input layer weights and biases
    #Layer1
    W1 = np.random.randn(784,128)*0.1 #np.random.randn(784,128) * np.sqrt(2/784) is better
    B1 = np.zeros((1,128))
    
    #Layer2
    W2 = np.random.randn(128,10)*0.1 
    B2 = np.zeros((1,10))
    
    return W1,B1,W2,B2



#### Phase 3: forward propogation


#Output activation

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def forward_propogation(X, W1, B1, W2, B2):
    
    #Layer1 process
    #MULTIPLY inputs by weights and add the bias
    Z1 = np.dot(X , W1) + B1  #
    
    #squish the result through ReLu
    #Hidden layer activation function
    A1 = np.maximum(0,Z1)
    
    #Layer2 process
    #multiply hidden layer's output by final weights
    Z2 = np.dot(A1, W2) + B2
    
    #Apply softmax to conver the result into probablities
    A2 = softmax(Z2)
    
    #return all values for backward propogation
    return Z1, A1, Z2, A2

### Phase 4: Loss function 
def compute_cross_entropy(Y, A2):   
    
    A2_clipped = np.clip(A2,1e-15 , 1.0 - 1e-15)
    loss = -np.sum(Y * np.log(A2_clipped)) / Y.shape[0] 
    return loss

def get_prediction(A2):
    return np.argmax(A2, axis=1)
    

def get_accuracy(predictions,y_labels):
    
    # Count how many times the prediction exactly matches the true label
    correct_guesses = np.sum(predictions == y_labels)
    # Divide by total number of images to get a decimal (e.g., 0.85)
    return correct_guesses / y_labels.size

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
##### Phase 5: BackPropogation

def relu_derivative(z):
    #Returns 1 if Z > 0, otherwise returns 0.
    return (z>0).astype(float)  ## .astype(float) turns True into 1.0 and False into 0.0


def backpropogation(Z1, A1, A2, W2, X, Y):
    
    ##1. output layer error
    m = X.shape[0]
    error_at_output = (A2 - Y) / m  

    # calculate gradient for W2 and B2 (output layer)
    dW2 = np.dot(A1.T, error_at_output)
    dB2 = np.sum(error_at_output, axis = 0, keepdims= True)
    
    ##2 Hidden layer Error (chain rule)   
    error_at_hidden_layer = np.dot(error_at_output,W2.T)
    dZ1 = error_at_hidden_layer*relu_derivative(Z1)
    
    #calculate gradient/blame score for W1 and B1
    dW1 = np.dot(X.T,dZ1)
    dB1 = np.sum(dZ1, axis = 0, keepdims = True)
    
    
    return dW1, dB1, dW2, dB2


##### Phase 6: Gradient Descent

def gradient_decent(X, Y, Y_labels, lr, epochs, patience=20, min_delta=1e-4):

    W1, B1, W2, B2 = initate_paramaters()

    best_loss = float("inf")
    best_params = (W1.copy(), B1.copy(), W2.copy(), B2.copy())
    epochs_without_improvement = 0

    for i in range(epochs):

        # forward_prop
        Z1, A1, Z2, A2 = forward_propogation(X, W1, B1, W2, B2)

        # loss for this epoch (needed every epoch for early stopping)
        loss = compute_cross_entropy(Y, A2)

        # ---------- early stopping check ----------
        if loss < best_loss - min_delta:
            best_loss = loss
            best_params = (W1.copy(), B1.copy(), W2.copy(), B2.copy())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {i}: no improvement for {patience} epochs "
                  f"(best loss {best_loss:.4f})")
            break

        if np.isnan(loss):
            print(f"Loss is NaN at epoch {i}, stopping.")
            break
        # ------------------------------------------

        # calculate the gradient
        dW1, dB1, dW2, dB2 = backpropogation(Z1, A1, A2, W2, X, Y)

        # update the weights and biases
        W1 = W1 - (lr * dW1)
        B1 = B1 - (lr * dB1)
        W2 = W2 - (lr * dW2)
        B2 = B2 - (lr * dB2)

        if i % 10 == 0:
            predictions = get_prediction(A2)
            accuracy = get_accuracy(predictions, Y_labels)
            print(f"Epoch {i:3} | Loss: {loss:8.4f} | Accuracy: {accuracy * 100:5.2f}%")

    # return the best weights, not necessarily the last ones
    W1, B1, W2, B2 = best_params
    return W1, B1, W2, B2
    
    


####Train & Test thy model
#### Phase 1: Data prepration
#--------------------------------------------------------------------------------------
def test_model(W1, B1, W2, B2):
    
    test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

    print(f"Test_I shape: {test_images.shape}") # Expected: (Num_items, Rows, Cols)
    print(f"Test_L shape: {test_labels.shape}")

    ##flatten images
    test_images_flattened  = test_images.reshape(test_images.shape[0], 784) / 255.0
    print(test_images_flattened.shape)
    
    
    #forward_prop/take a guess
    Z1, A1, Z2, A2 = forward_propogation(test_images_flattened, W1, B1, W2, B2)



    predictions = get_prediction(A2)
    y = test_labels.astype(int)
    n = len(y)
    
    ####AI Generated
    # ---------- 1. Overall metrics ----------
    correct_mask = (predictions == y)
    accuracy = get_accuracy(predictions, y)
    mean_loss = compute_cross_entropy(one_hot_encode(y), A2) / n   # per-sample loss

    print("=" * 62)
    print("OVERALL")
    print("=" * 62)
    print(f"Test samples        : {n}")
    print(f"Correct / Wrong     : {correct_mask.sum()} / {(~correct_mask).sum()}")
    print(f"Accuracy            : {accuracy * 100:.2f}%")
    print(f"Mean cross-entropy  : {mean_loss:.4f}   (untrained model ~ {np.log(10):.4f})")
    print(f"Baselines           : random guess = 10.00%, "
          f"always-predict-most-common = {np.bincount(y).max() / n * 100:.2f}%")
    print(f"NaNs in output?     : {np.isnan(A2).any()}")

    # ---------- 2. Confusion matrix (rows = true, cols = predicted) ----------
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, predictions), 1)

    print("\n" + "=" * 62)
    print("CONFUSION MATRIX (rows = true digit, cols = predicted digit)")
    print("=" * 62)
    print("      " + "".join(f"{d:6d}" for d in range(10)))
    for d in range(10):
        print(f"  {d}   " + "".join(f"{cm[d, j]:6d}" for j in range(10)))

    # ---------- 3. Per-class precision / recall / F1 ----------
    tp = np.diag(cm)
    support = cm.sum(axis=1)       # how many true examples of each digit
    predicted = cm.sum(axis=0)     # how many times each digit was predicted
    recall = tp / np.maximum(support, 1)
    precision = tp / np.maximum(predicted, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)

    print("\n" + "=" * 62)
    print("PER-CLASS METRICS")
    print("=" * 62)
    print(f"{'Digit':>5} {'Support':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for d in range(10):
        print(f"{d:>5} {support[d]:>8} {precision[d]*100:>9.2f}% {recall[d]*100:>7.2f}% {f1[d]*100:>7.2f}%")
    print(f"{'Macro':>5} {n:>8} {precision.mean()*100:>9.2f}% {recall.mean()*100:>7.2f}% {f1.mean()*100:>7.2f}%")
    print(f"Weakest digit (recall): {np.argmin(recall)} at {recall.min()*100:.2f}%")

    # ---------- 4. Most common mistakes ----------
    off_diag = cm.copy()
    np.fill_diagonal(off_diag, 0)
    top_idx = np.argsort(off_diag, axis=None)[::-1][:5]

    print("\n" + "=" * 62)
    print("TOP 5 CONFUSIONS")
    print("=" * 62)
    for idx in top_idx:
        t, p = divmod(idx, 10)
        print(f"True {t} predicted as {p}: {off_diag[t, p]} times")

    # ---------- 5. Confidence analysis ----------
    confidence = A2.max(axis=1)
    print("\n" + "=" * 62)
    print("CONFIDENCE")
    print("=" * 62)
    print(f"Avg confidence when correct : {confidence[correct_mask].mean() * 100:.2f}%")
    if (~correct_mask).any():
        print(f"Avg confidence when WRONG   : {confidence[~correct_mask].mean() * 100:.2f}%")
        print(f"Wrong with >90% confidence  : {np.sum((~correct_mask) & (confidence > 0.9))}")
        wrong_idx = np.where(~correct_mask)[0]
        worst = wrong_idx[np.argsort(confidence[wrong_idx])[::-1][:5]]
        print("Most confidently wrong (image index: true -> pred @ confidence):")
        for i in worst:
            print(f"   #{i}: {y[i]} -> {predictions[i]} @ {confidence[i]*100:.1f}%")





def run_model():

    ## Read the files
    train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    
    # See the shapes of your datasets
    print(f"Images shape: {train_images.shape}") # Expected: (Num_items, Rows, Cols)
    print(f"Labels shape: {train_labels.shape}") # Expected: (Num_items,)
    
    ##flatten images
    train_images_flattened = train_images.reshape(train_images.shape[0], 784) / 255.0
    print(f"Flattened Images shape: {train_images_flattened.shape}") 

    

    fig, axes = plt.subplots(1,5,figsize = (10,2)) 
    for i in range(5):
        axes[i].imshow(train_images[i], cmap ='gray')
        axes[i].axis('off')
    plt.show()
    
    
    Y =  one_hot_encode(train_labels)
    LR = 0.1 #learning_rate
    epochs = 1000
    
    W1, B1, W2, B2 = gradient_decent(train_images_flattened, Y, train_labels, LR, epochs)
    
    test_model(W1, B1, W2, B2)
    

if __name__ == "__main__":
    run_model()