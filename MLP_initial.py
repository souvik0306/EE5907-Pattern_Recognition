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

# Initialize weights and biases
np.random.seed(42)  # For reproducibility
W1 = np.random.randn(2, 3)  # Weights between input and hidden layer
b1 = np.random.randn(3)     # Biases for hidden layer
W2 = np.random.randn(3, 1)  # Weights between hidden and output layer
b2 = np.random.randn(1)     # Bias for output layer

# Forward pass function
def forward_pass(X):
    # Hidden layer computation
    z1 = X @ W1 + b1
    a1 = relu(z1)
    
    # Output layer computation
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)  # Binary classification output between 0 and 1
    
    return a2

# Compute initial predictions and classification accuracy
predictions = forward_pass(X)
predicted_labels = (predictions > 0.5).astype(int).flatten()
accuracy = np.mean(predicted_labels == T)
print(f'Initial Classification Accuracy: {accuracy * 100:.2f}%')

# Plot the decision boundary
def plot_decision_boundary(X, T, forward_pass):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = forward_pass(grid).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Initial MLP')
    plt.show()

# Plot the decision boundary of the MLP
plot_decision_boundary(X, T, forward_pass)
