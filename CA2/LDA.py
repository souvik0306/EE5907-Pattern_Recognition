import numpy as np
import matplotlib.pyplot as plt
from dataset import load_data  # Ensure dataset.py is in the same directory
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load dataset
cmu_data_folder = 'CA2/PIE'  # Replace with your dataset path
selfie_folder = 'CA2/PIE/Selfie' 
train_images, train_labels, test_images, test_labels = load_data(cmu_data_folder, selfie_folder)

# Randomly sample 500 images (CMU PIE + Selfies)
np.random.seed(42)
sample_indices = np.random.choice(len(train_images), 500, replace=False)
sample_images = train_images[sample_indices]
sample_labels = train_labels[sample_indices]

# Apply LDA
def apply_lda(data, labels, n_components):
    lda = LDA(n_components=n_components)
    transformed_data = lda.fit_transform(data, labels)
    return transformed_data, lda

# 2D Visualization
def visualize_lda_2d(data, labels, highlighted_idx=None):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    if highlighted_idx is not None:
        plt.scatter(data[highlighted_idx, 0], data[highlighted_idx, 1], color='red', label='Selfie')
    plt.xlabel('LD 1')
    plt.ylabel('LD 2')
    plt.legend()
    plt.title("2D LDA Visualization")
    plt.colorbar(scatter, label="Labels")
    plt.show()

# 3D Visualization
def visualize_lda_3d(data, labels, highlighted_idx=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', alpha=0.7)
    if highlighted_idx is not None:
        ax.scatter(data[highlighted_idx, 0], data[highlighted_idx, 1], data[highlighted_idx, 2], color='red', label='Selfie')
    ax.set_xlabel('LD 1')
    ax.set_ylabel('LD 2')
    ax.set_zlabel('LD 3')
    plt.title("3D LDA Visualization")
    plt.colorbar(scatter, label="Labels")
    plt.legend()
    plt.show()

# Highlighted indices for your photos in the sample
highlighted_indices = [idx for idx, label in enumerate(sample_labels) if label == max(sample_labels)]

# Apply LDA for visualization
lda_data_2d, _ = apply_lda(sample_images, sample_labels, n_components=2)
visualize_lda_2d(lda_data_2d, sample_labels, highlighted_indices)

lda_data_3d, _ = apply_lda(sample_images, sample_labels, n_components=3)
visualize_lda_3d(lda_data_3d, sample_labels, highlighted_indices)

import numpy as np
from dataset import load_data  # Ensure dataset.py is in the same directory
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_images, train_labels, test_images, test_labels = load_data(cmu_data_folder, selfie_folder)

# Apply LDA
def apply_lda(data, labels, n_components):
    lda = LDA(n_components=n_components)
    transformed_data = lda.fit_transform(data, labels)
    return transformed_data, lda

# Nearest Neighbor Classifier
def nearest_neighbor_classifier(train_data, train_labels, test_data, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, predictions

# Function to run LDA classification
def lda_classification(train_images, train_labels, test_images, test_labels, dimensions):
    results = {}
    for n_components in dimensions:
        # Apply LDA to train and test sets
        lda_data_train, lda = apply_lda(train_images, train_labels, n_components)
        lda_data_test = lda.transform(test_images)
        
        # Perform classification
        accuracy, predictions = nearest_neighbor_classifier(lda_data_train, train_labels, lda_data_test, test_labels)
        
        # Separate accuracy for CMU PIE and selfie test images
        cmu_pie_indices = [i for i, label in enumerate(test_labels) if label != max(train_labels)]
        selfie_indices = [i for i, label in enumerate(test_labels) if label == max(train_labels)]
        
        cmu_pie_accuracy = accuracy_score(test_labels[cmu_pie_indices], predictions[cmu_pie_indices])
        selfie_accuracy = accuracy_score(test_labels[selfie_indices], predictions[selfie_indices])
        
        results[n_components] = {"overall": accuracy, "cmu_pie": cmu_pie_accuracy, "selfie": selfie_accuracy}
    
    return results

# Define LDA dimensions
lda_dimensions = [2, 3, 9]

# Run LDA classification and print results
results = lda_classification(train_images, train_labels, test_images, test_labels, lda_dimensions)
for n_components, accuracy_dict in results.items():
    print(f"LDA Dimensionality {n_components}:")
    print(f"  Overall Accuracy = {accuracy_dict['overall']:.2f}")
    print(f"  CMU PIE Accuracy = {accuracy_dict['cmu_pie']:.2f}")
    print(f"  Selfie Accuracy = {accuracy_dict['selfie']:.2f}")


