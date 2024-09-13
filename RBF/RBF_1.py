import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

# Combine data and labels
X = np.concatenate((X1, X2))
T = np.array([0] * len(X1) + [1] * len(X2))  # Ground-truth labels: 0 for Class 1, 1 for Class 2

# Plot the data
c1 = ['blue'] * len(X1)  # Class 1
c2 = ['red'] * len(X2) # Class 2
color = np.concatenate((c1, c2))
plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Loaded Binary Class Data')
plt.show()

# Gaussian RBF function
def rbf(x, center, sigma=1):
    return np.exp(-np.linalg.norm(x - center, axis=1) ** 2 / (2 * sigma ** 2))

# Randomly set RBF centers within the range of the data
np.random.seed(42)  # For reproducibility
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()

centers = np.random.rand(6, 2) * [x_max - x_min, y_max - y_min] + [x_min, y_min]

# Plot the data with RBF centers
plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='green', s=100, label='RBF Centers')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('RBF Centers within Data Range')
plt.legend()
plt.show()

# Construct the RBF matrix
def construct_rbf_matrix(X, centers, sigma=1):
    G = np.zeros((X.shape[0], centers.shape[0]))
    for i, center in enumerate(centers):
        G[:, i] = rbf(X, center, sigma)
    return G

# Construct the RBF matrix for the data
G = construct_rbf_matrix(X, centers, sigma=1)

# Compute the weights using Least Squares Estimation
W = np.linalg.pinv(G) @ T

# Forward pass function for RBF network
def rbf_network(X, centers, W, sigma=1):
    G = construct_rbf_matrix(X, centers, sigma)
    return G @ W

# Predict and calculate classification accuracy
predictions = rbf_network(X, centers, W)
predicted_labels = (predictions > 0.5).astype(int)
accuracy = np.mean(predicted_labels == T)

print(f'RBF Network Classification Accuracy: {accuracy * 100:.2f}%')

# Plot the decision boundary of the RBF network
def plot_decision_boundary_rbf(X, T, centers, W, sigma=1):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = rbf_network(grid, centers, W, sigma).reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'], linestyles='--')
    plt.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=1, linestyles='--')
    plt.scatter(X[:, 0], X[:, 1], c=T, cmap='bwr', edgecolor='k')
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='green', s=100, label='RBF Centers')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary of Inital RBF Network with Accuracy = {accuracy*100}%')
    plt.legend()
    plt.show()

# Plot the decision boundary of the RBF network
plot_decision_boundary_rbf(X, T, centers, W, sigma=1)
