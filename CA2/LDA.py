import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os
import cv2

def load_images_from_folder(folder_path, img_size=(32, 32), max_folders=25):
    images, labels = [], []
    label = 0
    subject_folders = sorted(os.listdir(folder_path))[:max_folders]
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
    selfie_images = []
    for img_file in os.listdir(selfie_folder):
        img_path = os.path.join(selfie_folder, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            selfie_images.append(img.flatten())
    return np.array(selfie_images)

def load_data(cmu_data_folder, selfie_folder, max_folders=25, random_seed=42):
    images, labels = load_images_from_folder(cmu_data_folder, max_folders=max_folders)
    selfie_images = prepare_selfie_images(selfie_folder)

    train_images, train_labels, test_images, test_labels = [], [], [], []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_images = images[label_indices]
        label_train, label_test = train_test_split(label_images, test_size=0.3, random_state=random_seed)
        train_images.extend(label_train)
        test_images.extend(label_test)
        train_labels.extend([label] * len(label_train))
        test_labels.extend([label] * len(label_test))

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    selfie_train, selfie_test = train_test_split(selfie_images, test_size=0.3, random_state=random_seed)
    selfie_train_labels = np.full(len(selfie_train), label + 1)
    selfie_test_labels = np.full(len(selfie_test), label + 1)

    train_images = np.vstack((train_images, selfie_train))
    train_labels = np.concatenate((train_labels, selfie_train_labels))
    test_images = np.vstack((test_images, selfie_test))
    test_labels = np.concatenate((test_labels, selfie_test_labels))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    return train_images, train_labels, test_images, test_labels

def apply_lda(data, labels, n_components):
    lda = LDA(n_components=n_components)
    transformed_data = lda.fit_transform(data, labels)
    return transformed_data, lda

def visualize_lda_2d(data, labels, highlighted_idx=None):
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']
    colors = plt.cm.tab20.colors

    for i, unique_label in enumerate(unique_labels):
        idx = labels == unique_label
        if unique_label == max(labels):
            plt.scatter(data[idx, 0], data[idx, 1], color='red', marker='o', label='Selfie', edgecolor='k')
        else:
            marker = markers[i % len(markers)]
            plt.scatter(data[idx, 0], data[idx, 1], color=colors[i % len(colors)], marker=marker, label=f"Class {unique_label}")

    plt.xlabel("LD 1")
    plt.ylabel("LD 2")
    plt.legend()
    plt.title("2D LDA Visualization")
    plt.show()

def visualize_lda_3d(data, labels, highlighted_idx=None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    unique_labels = np.unique(labels)
    markers = ['o', 's', '^', 'P', '*', 'X', 'D']
    colors = plt.cm.tab20.colors

    for i, unique_label in enumerate(unique_labels):
        idx = labels == unique_label
        if unique_label == max(labels):
            ax.scatter(data[idx, 0], data[idx, 1], data[idx, 2], color='red', marker='o', label='Selfie', edgecolor='k')
        else:
            marker = markers[i % len(markers)]
            ax.scatter(data[idx, 0], data[idx, 1], data[idx, 2], color=colors[i % len(colors)], marker=marker, label=f"Class {unique_label}")

    ax.set_xlabel("LD 1")
    ax.set_ylabel("LD 2")
    ax.set_zlabel("LD 3")
    plt.legend()
    plt.title("3D LDA Visualization")
    plt.show()

def nearest_neighbor_classifier(train_data, train_labels, test_data, test_labels):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_data, train_labels)
    predictions = knn.predict(test_data)
    accuracy = accuracy_score(test_labels, predictions)
    return accuracy, predictions

def lda_classification(train_images, train_labels, test_images, test_labels, dimensions):
    results = {}
    for n_components in dimensions:
        lda_data_train, lda = apply_lda(train_images, train_labels, n_components)
        lda_data_test = lda.transform(test_images)

        accuracy, predictions = nearest_neighbor_classifier(lda_data_train, train_labels, lda_data_test, test_labels)

        cmu_pie_indices = [i for i, label in enumerate(test_labels) if label != max(train_labels)]
        selfie_indices = [i for i, label in enumerate(test_labels) if label == max(train_labels)]

        cmu_pie_accuracy = accuracy_score(test_labels[cmu_pie_indices], predictions[cmu_pie_indices])
        selfie_accuracy = accuracy_score(test_labels[selfie_indices], predictions[selfie_indices])

        results[n_components] = {"overall": accuracy, "cmu_pie": cmu_pie_accuracy, "selfie": selfie_accuracy}

    return results

def main():
    seed = 23
    cmu_data_folder = "CA2/PIE"
    selfie_folder = "CA2/PIE/Selfie"
    train_images, train_labels, test_images, test_labels = load_data(cmu_data_folder, selfie_folder, random_seed=seed)

    scaler = StandardScaler()
    train_images = scaler.fit_transform(train_images)
    test_images = scaler.transform(test_images)

    lda_dimensions = [2, 3, 9]
    results = lda_classification(train_images, train_labels, test_images, test_labels, lda_dimensions)
    for n_components, accuracy_dict in results.items():
        print(f"LDA Dimensionality {n_components}:")
        print(f"  Overall Accuracy = {accuracy_dict['overall']:.4f}")
        print(f"  CMU PIE Accuracy = {accuracy_dict['cmu_pie']:.4f}")
        print(f"  Selfie Accuracy = {accuracy_dict['selfie']:.4f}")

    sample_indices = np.random.choice(len(train_images), 500, replace=False)
    sample_images = train_images[sample_indices]
    sample_labels = train_labels[sample_indices]

    highlighted_indices = [idx for idx, label in enumerate(sample_labels) if label == max(sample_labels)]

    lda_data_2d, _ = apply_lda(sample_images, sample_labels, n_components=2)
    visualize_lda_2d(lda_data_2d, sample_labels, highlighted_indices)

    lda_data_3d, _ = apply_lda(sample_images, sample_labels, n_components=3)
    visualize_lda_3d(lda_data_3d, sample_labels, highlighted_indices)

if __name__ == "__main__":
    main()