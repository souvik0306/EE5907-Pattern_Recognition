import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# Labels for the classes
c1 = ['red'] * len(X1)  # Class 1
c2 = ['blue'] * len(X2) # Class 2

# Combine data
X = np.concatenate((X1, X2))
color = np.concatenate((c1, c2))

# Ground-truth labels: 0 for Class 1, 1 for Class 2
T = np.array([0] * len(X1) + [1] * len(X2))

# Plot the data
plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Loaded Binary Class Data')
# plt.show()

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights randomly
np.random.seed(42)
weights_input_hidden = np.random.normal(0, 1, (2, 3))  # 2 input neurons, 3 hidden neurons
weights_hidden_output = np.random.normal(0, 1, (3, 1))  # 3 hidden neurons, 1 output neuron

# Forward pass function
def forward_pass(X):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)
    return output_layer_output, hidden_layer_output

# Decision boundary plotting function
def plot_decision_boundary(X, y, weights_input_hidden, weights_hidden_output):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions, _ = forward_pass(grid)
    Z = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 10), alpha=0.6, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary of Initial MLP')
    plt.show()

# Initial decision boundary
plot_decision_boundary(X, T, weights_input_hidden, weights_hidden_output)

# Compute initial classification accuracy
initial_predictions, _ = forward_pass(X)
initial_accuracy = np.mean((initial_predictions > 0.5).astype(int) == T) * 100
print(f'Initial Classification Accuracy: {initial_accuracy:.2f}%')
