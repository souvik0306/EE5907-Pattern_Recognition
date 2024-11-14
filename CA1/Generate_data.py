import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io

N1 = 50
N2 = 50
K = 2
sigma = 2

mean1 = (10, 12)
cov1 = [[sigma, 0], [0, sigma]]
X1 = np.random.multivariate_normal(mean1, cov1, N1)
c1 = ['red'] * len(X1) #Class 1

mean2 = (12, 14)
cov2 = [[sigma, 0], [0, sigma]]
X2 = np.random.multivariate_normal(mean2, cov2, N2)
c2 = ['blue'] * len(X2) #Class 2

X = np.concatenate((X1, X2))
color = np.concatenate((c1, c2))

T = []
for n in range(0, len(X)):
    if (n < len(X1)):
        T.append(0)
    else:
        T.append(1)

plt.scatter(X[:, 0], X[:, 1], marker = 'o', c = color)
plt.show()

# np.save('class1.npy', X1)
# np.save('class2.npy', X2)
# io.savemat('class1.mat', {'class1': X1})
# io.savemat('class2.mat', {'class2': X2})