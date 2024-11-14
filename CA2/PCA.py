import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess dataset
def load_images_from_folder(folder_path, img_size=(32, 32), max_folders=25):
    """
    Loads images from the first 'max_folders' folders in the specified directory.
    """
    images, labels = [], []
    label = 0
    subject_folders = sorted(os.listdir(folder_path))[:max_folders]  # Take only the first 'max_folders'
    for subject_folder in subject_folders:
        subject_path = os.path.join(folder_path, subject_folder)
        if os.path.isdir(subject_path):
            for img_file in os.listdir(subject_path):
                img_path = os.path.join(subject_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    images.append(img.flatten())
                    labels.append(label)
            label += 1
    return np.array(images), np.array(labels)

def prepare_selfie_images(selfie_folder, img_size=(32, 32)):
    """
    Loads all images from the specified selfie folder.
    """
    selfie_images = []
    for img_file in os.listdir(selfie_folder):
        img_path = os.path.join(selfie_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            selfie_images.append(img.flatten())
    return np.array(selfie_images)

def load_data(cmu_data_folder, selfie_folder, max_folders=25, random_seed=42):
    """
    Loads data from the CMU PIE dataset and selfie images.
    Only the first 'max_folders' folders from CMU PIE are loaded.
    """
    images, labels = load_images_from_folder(cmu_data_folder, max_folders=max_folders)
    selfie_images = prepare_selfie_images(selfie_folder)

    train_images, train_labels, test_images, test_labels = [], [], [], []
    unique_labels = np.unique(labels)

    # Train-test split for CMU PIE dataset
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_images = images[label_indices]
        label_train, label_test = train_test_split(
            label_images, test_size=0.3, random_state=random_seed
        )
        train_images.extend(label_train)
        test_images.extend(label_test)
        train_labels.extend([label] * len(label_train))
        test_labels.extend([label] * len(label_test))

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    # Train-test split for selfies
    selfie_train, selfie_test = train_test_split(
        selfie_images, test_size=0.3, random_state=random_seed
    )
    selfie_train_labels = np.full(len(selfie_train), label + 1)
    selfie_test_labels = np.full(len(selfie_test), label + 1)

    train_images = np.vstack((train_images, selfie_train))
    train_labels = np.concatenate((train_labels, selfie_train_labels))
    test_images = np.vstack((test_images, selfie_test))
    test_labels = np.concatenate((test_labels, selfie_test_labels))

    # Normalize data for PCA
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

# PCA and visualization functions
def apply_pca(data, n_components, random_seed=42):
    pca = PCA(n_components=n_components, random_state=random_seed)
    transformed_data = pca.fit_transform(data)
    return transformed_data, pca

def visualize_pca_2d(data, labels, highlighted_idx=None):
    """
    Visualizes 2D PCA data with different colors and shapes for each class.
    Highlights selfie images in red dots.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']  # Marker styles for different classes
    colors = plt.cm.tab20.colors  # Use a color palette

    for i, unique_label in enumerate(unique_labels):
        idx = labels == unique_label
        if unique_label == max(labels):  # Selfie class (last label)
            plt.scatter(data[idx, 0], data[idx, 1], color='red', marker='o', label='Selfie', edgecolor='k')
        else:
            marker = markers[i % len(markers)]  # Cycle through marker styles
            plt.scatter(data[idx, 0], data[idx, 1], color=colors[i % len(colors)], marker=marker, label=f"Class {unique_label}")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.title("2D PCA Visualization")
    plt.show()

def visualize_pca_3d(data, labels, highlighted_idx=None):
    """
    Visualizes 3D PCA data with different colors and shapes for each class.
    Highlights selfie images in red dots.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']  # Marker styles for different classes
    colors = plt.cm.tab20.colors  # Use a color palette

    for i, unique_label in enumerate(unique_labels):
        idx = labels == unique_label
        if unique_label == max(labels):  # Selfie class (last label)
            ax.scatter(data[idx, 0], data[idx, 1], data[idx, 2], color='red', marker='o', label='Selfie', edgecolor='k')
        else:
            marker = markers[i % len(markers)]  # Cycle through marker styles
            ax.scatter(
                data[idx, 0], data[idx, 1], data[idx, 2],
                color=colors[i % len(colors)], marker=marker, label=f"Class {unique_label}"
            )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.legend()
    plt.title("3D PCA Visualization")
    plt.show()

def visualize_eigenfaces(pca, img_shape=(32, 32), n_components=3):
    plt.figure(figsize=(10, 4))
    for i in range(n_components):
        eigenface = pca.components_[i].reshape(img_shape)
        plt.subplot(1, n_components, i + 1)
        plt.imshow(eigenface, cmap="gray")
        plt.title(f"Eigenface {i + 1}")
        plt.axis("off")
    plt.suptitle("Top 3 Eigenfaces")
    plt.show()

# Nearest Neighbor Classifier
def nearest_neighbor_classifier(train_data, train_labels, test_data, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, predictions

# PCA classification function
def pca_classification(train_images, train_labels, test_images, test_labels, dimensions):
    results = {}
    for n_components in dimensions:
        pca_data_train, pca = apply_pca(train_images, n_components)
        pca_data_test = pca.transform(test_images)
        accuracy, predictions = nearest_neighbor_classifier(
            pca_data_train, train_labels, pca_data_test, test_labels
        )

        cmu_pie_indices = [i for i, label in enumerate(test_labels) if label != max(train_labels)]
        selfie_indices = [i for i, label in enumerate(test_labels) if label == max(train_labels)]

        cmu_pie_accuracy = accuracy_score(test_labels[cmu_pie_indices], predictions[cmu_pie_indices])
        selfie_accuracy = accuracy_score(test_labels[selfie_indices], predictions[selfie_indices])

        results[n_components] = {
            "overall": accuracy,
            "cmu_pie": cmu_pie_accuracy,
            "selfie": selfie_accuracy,
        }
    return results

# Main execution
if __name__ == "__main__":
    seed = 8
    cmu_data_folder = "CA2/PIE"
    selfie_folder = "CA2/PIE/Selfie"
    train_images, train_labels, test_images, test_labels = load_data(
        cmu_data_folder, selfie_folder, random_seed=seed
    )

    pca_dimensions = [40, 80, 200]
    results = pca_classification(
        train_images, train_labels, test_images, test_labels, pca_dimensions
    )
    for n_components, accuracy_dict in results.items():
        print(f"PCA Dimensionality {n_components}:")
        print(f"  Overall Accuracy = {accuracy_dict['overall']:.4f}")
        print(f"  CMU PIE Accuracy = {accuracy_dict['cmu_pie']:.4f}")
        print(f"  Selfie Accuracy = {accuracy_dict['selfie']:.4f}")

    # Visualize PCA
    sample_indices = np.random.choice(len(train_images), 500, replace=False)
    sample_images = train_images[sample_indices]
    sample_labels = train_labels[sample_indices]
    highlighted_indices = [idx for idx, label in enumerate(sample_labels) if label == max(sample_labels)]

    pca_data_2d, pca_model_2d = apply_pca(sample_images, n_components=2, random_seed=seed)
    visualize_pca_2d(pca_data_2d, sample_labels, highlighted_indices)

    pca_data_3d, pca_model_3d = apply_pca(sample_images, n_components=3, random_seed=seed)
    visualize_pca_3d(pca_data_3d, sample_labels, highlighted_indices)

    visualize_eigenfaces(pca_model_3d)
