import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# Combine data and labels
X = np.concatenate((X1, X2))
T = np.array([0] * len(X1) + [1] * len(X2))  # Ground-truth labels: 0 for Class 1, 1 for Class 2

# Define activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the forward pass
def forward_pass(X, W1, b1, W2, b2):
    # Hidden layer computation
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Output layer computation
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)  # Binary classification output between 0 and 1
    
    return a2, a1, z1

# Define the backward pass
def backward_pass(X, T, a1, a2, z1, W2):
    m = len(T)
    
    # Output layer error using binary cross-entropy loss derivative
    dL_dz2 = a2 - T.reshape(-1, 1)

    # Gradients for W2 and b2
    dL_dW2 = a1.T @ dL_dz2 / m
    dL_db2 = np.sum(dL_dz2, axis=0) / m
    
    # Hidden layer error
    dz2_da1 = W2
    dL_da1 = dL_dz2 @ dz2_da1.T
    da1_dz1 = relu_derivative(z1)
    dL_dz1 = dL_da1 * da1_dz1

    # Gradients for W1 and b1
    dL_dW1 = X.T @ dL_dz1 / m
    dL_db1 = np.sum(dL_dz1, axis=0) / m
    
    return dL_dW1, dL_db1, dL_dW2, dL_db2

# Initialize weights and biases with random initialization
W1 = np.random.randn(2, 3)  # Weights between input and hidden layer
b1 = np.random.randn(3)     # Biases for hidden layer
W2 = np.random.randn(3, 1)  # Weights between hidden and output layer
b2 = np.random.randn(1)     # Bias for output layer

# Training parameters
learning_rate = 0.015
epochs = 8000
target_accuracy = 0.8  # Set target accuracy threshold
reached_threshold = False

# List to store accuracy for each epoch
accuracies = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    a2, a1, z1 = forward_pass(X, W1, b1, W2, b2)
    
    # Backward pass
    dL_dW1, dL_db1, dL_dW2, dL_db2 = backward_pass(X, T, a1, a2, z1, W2)
    
    # Update weights and biases
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2

    # Compute predictions and classification accuracy
    predictions = (a2 > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == T)
    accuracies.append(accuracy)  # Store accuracy for this epoch
    
    # Check if target accuracy threshold is reached
    if accuracy >= target_accuracy:
        print(f'Target Accuracy of {target_accuracy * 100:.2f}% reached at epoch {epoch + 1}')
        reached_threshold = True
        break

# Check if threshold was not reached within the specified epochs
if not reached_threshold:
    print(f'Target accuracy of {target_accuracy * 100:.2f}% was not reached within {epochs} epochs.')

# Print final accuracy and mean accuracy
final_accuracy = accuracies[-1] if accuracies else 0
mean_accuracy = np.mean(accuracies)
print(f'Final Classification Accuracy: {final_accuracy * 100:.2f}%')
print(f'Mean Classification Accuracy over all epochs: {mean_accuracy * 100:.2f}%')

# Plot the decision boundary for the updated MLP
def plot_decision_boundary(X, T, forward_pass, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs, _, _ = forward_pass(grid, W1, b1, W2, b2)
    probs = probs.reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary of MLP using Backprop with Final Accuracy = {final_accuracy*100:.2f}%')
    plt.show()

# Plot the updated decision boundary if accuracy is above the threshold
if final_accuracy >= target_accuracy:
    plot_decision_boundary(X, T, forward_pass, W1, b1, W2, b2)
else:
    print(f'Target accuracy of {target_accuracy * 100:.2f}% was not reached.')
