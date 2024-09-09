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
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=color)
# plt.xlabel('Class 1')
# plt.ylabel('Class 2')
# plt.title('Scatter Plot of Loaded Binary Class Data')
# plt.show()

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.random.randn(self.output_size)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Initialize the MLP
mlp = MLP(input_size=2, hidden_size=3, output_size=1)

# Function to plot decision boundary
def plot_decision_boundary(X, T, mlp):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4,  colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=color, marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary of Trained MLP with Accuracy')
    plt.show()

# Plot decision boundary
plot_decision_boundary(X, T, mlp)

# Calculate initial accuracy
initial_predictions = mlp.predict(X)
initial_accuracy = np.mean(initial_predictions.flatten() == T)
print(f"Initial classification accuracy: {initial_accuracy:.2%}")