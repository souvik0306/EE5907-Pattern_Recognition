### **Technical Abstract**

This project explores the application of various machine learning techniques for face image classification tasks. The techniques include:

- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Support Vector Machines (SVM)
- Convolutional Neural Networks (CNN)

Using a dataset comprising CMU PIE images and a distinct selfie class, the study investigates dimensionality reduction, feature extraction, and classification performance across multiple approaches.

#### **Principal Component Analysis (PCA)**

PCA was employed to reduce the dimensionality of 1024-dimensional face images, enabling visualization in 2D and 3D while identifying dominant features through eigenfaces. The classification accuracy of a nearest-neighbour classifier improved with higher PCA dimensions, reaching 96.33% at 200 components.

#### **Linear Discriminant Analysis (LDA)**

LDA further enhanced class separability by projecting data into maximally discriminative subspaces. While accuracy improved with higher dimensions, the selfie class posed challenges due to overlapping features.

#### **Support Vector Machines (SVM)**

A linear SVM was used to evaluate the effect of raw and PCA-reduced data on classification performance, achieving the highest accuracy of 99.39% with raw data. PCA-reduced vectors preserved classification performance, demonstrating the utility of dimensionality reduction.

#### **Convolutional Neural Networks (CNN)**

Finally, a CNN was trained for multi-class face image classification, leveraging convolutional layers and max pooling to extract spatial hierarchies, achieving a test accuracy of 96.83%.

The study highlights the trade-offs between dimensionality reduction, feature extraction, and model complexity, providing insights into their respective strengths and limitations for face recognition tasks.