import numpy as np
import matplotlib.pyplot as plt

# Load the data from npy files
X1 = np.load('class1.npy')
X2 = np.load('class2.npy')

print(X1,X2)
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
plt.xlabel('Class 1')
plt.ylabel('Class 2')
plt.title('Scatter Plot of Loaded Binary Class Data')
plt.show()
