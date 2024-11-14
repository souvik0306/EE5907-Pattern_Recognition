import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# Combine data and labels
X = np.concatenate((X1, X2))
T = np.array([0] * len(X1) + [1] * len(X2))  # Ground-truth labels: 0 for Class 1, 1 for Class 2

# Plot the data
c1 = ['red'] * len(X1)  # Class 1
c2 = ['blue'] * len(X2) # Class 2
color = np.concatenate((c1, c2))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter Plot of Loaded Binary Class Data')
# plt.show()

# Define activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Forward pass function
def forward_pass(X, W1, b1, W2, b2):
    # Hidden layer computation
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Output layer computation
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)  # Binary classification output between 0 and 1
    
    return a2

# Find the best weights and biases over 50 runs
best_accuracy = 0
best_W1, best_b1, best_W2, best_b2 = None, None, None, None

for i in range(3000,9900):
    np.random.seed(i)
    # Initialize weights and biases
    W1 = np.random.randn(2, 3)  # Weights between input and hidden layer
    b1 = np.random.randn(3)     # Biases for hidden layer
    W2 = np.random.randn(3, 1)  # Weights between hidden and output layer
    b2 = np.random.randn(1)     # Bias for output layer

    # Compute predictions and classification accuracy
    predictions = forward_pass(X, W1, b1, W2, b2)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(predicted_labels == T)

    # Check if the current accuracy is the best
    if accuracy > best_accuracy and accuracy==0.63:
        best_accuracy = accuracy
        print(i)
        best_W1, best_b1, best_W2, best_b2 = W1, b1, W2, b2

print(f'Best Classification Accuracy: {best_accuracy * 100:.2f}%')

# Plot the decision boundary for the best MLP
def plot_decision_boundary(X, T, forward_pass, W1, b1, W2, b2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = forward_pass(grid, W1, b1, W2, b2).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary for the Inital MLP with Accuracy = {best_accuracy*100}%')
    plt.show()

# Plot the decision boundary of the best MLP
plot_decision_boundary(X, T, forward_pass, best_W1, best_b1, best_W2, best_b2)
