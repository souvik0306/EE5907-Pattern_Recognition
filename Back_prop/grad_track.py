import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# Combine data and labels
X = np.concatenate((X1, X2))
T = np.array([0] * len(X1) + [1] * len(X2))  # Ground-truth labels: 0 for Class 1, 1 for Class 2

# Activation functions and their derivatives
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Loss function: Binary cross-entropy
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # To prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Forward pass function
def forward_pass(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward pass function (Backpropagation)
def backward_pass(X, T, z1, a1, z2, a2, W1, W2):
    # Output layer error and gradient
    output_error = a2 - T.reshape(-1, 1)
    dW2 = a1.T @ output_error
    db2 = np.sum(output_error, axis=0)

    # Hidden layer error and gradient
    hidden_error = (output_error @ W2.T) * tanh_derivative(z1)
    dW1 = X.T @ hidden_error
    db1 = np.sum(hidden_error, axis=0)

    return dW1, db1, dW2, db2

# Training parameters
learning_rate = 0.005
epochs = 500  # Increased epochs
best_accuracy = 0

# Initialize weights and biases using He initialization
# np.random.seed(34)  # For reproducibility
W1 = np.random.randn(2, 3)   # He initialization for hidden layer
b1 = np.zeros(3)     # Biases for hidden layer
W2 = np.random.randn(3, 1) # He initialization for output layer
b2 = np.zeros(1)     # Bias for output layer

accuracies = []
best_accuracies = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    z1, a1, z2, a2 = forward_pass(X, W1, b1, W2, b2)
    
    # Compute loss
    loss = binary_cross_entropy(T, a2)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward_pass(X, T, z1, a1, z2, a2, W1, W2)
    
    # Update weights and biases using gradient descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Calculate accuracy
    predictions = (a2 > 0.5).astype(int).flatten()
    accuracy = np.mean(predictions == T)
    accuracies.append(accuracy)
    
    # Update best accuracy if current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_W1, best_b1, best_W2, best_b2 = W1, b1, W2, b2

    # Record the best accuracy in each interval of 100 epochs
    if (epoch + 1) % 100 == 0:
        best_accuracies.append(best_accuracy)

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%')

print(f'Best Classification Accuracy after training: {best_accuracy * 100:.2f}%')

# Compute mean accuracy for every 100 epochs
mean_accuracies = [np.mean(accuracies[i:i+100]) for i in range(0, len(accuracies), 100)]

# Print the final mean accuracy
print(f'Final Mean Classification Accuracy: {np.mean(accuracies[-100:]) * 100:.2f}%')

# Plot mean and best accuracy curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(mean_accuracies) + 1), mean_accuracies, marker='o', label='Mean Accuracy')
plt.plot(range(1, len(best_accuracies) + 1), best_accuracies, marker='x', linestyle='--', label='Best Accuracy')
plt.xlabel('Epochs (in hundreds)')
plt.ylabel('Accuracy')
plt.title('Mean vs Best Accuracy Curve for Every 100 Epochs')
plt.grid(True)
plt.legend()
plt.show()

# Plot the decision boundary for the best MLP
def plot_decision_boundary(X, T, forward_pass, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    _, _, _, probs = forward_pass(grid, W1, b1, W2, b2)
    probs = probs.reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary of MLP using Backprop with Best Accuracy = {best_accuracy*100:.2f}%')
    plt.show()

# Plot the decision boundary of the best MLP
plot_decision_boundary(X, T, forward_pass, best_W1, best_b1, best_W2, best_b2)
