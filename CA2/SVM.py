import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import cv2
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, img_size=(32, 32)):
    images = []
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img.flatten())
    return np.array(images)

def load_cmu_pie_data(cmu_data_folder):
    images, labels = [], []
    label = 0
    for subject_folder in os.listdir(cmu_data_folder):
        subject_path = os.path.join(cmu_data_folder, subject_folder)
        if os.path.isdir(subject_path):
            subject_images = load_images_from_folder(subject_path)
            train_imgs, test_imgs = train_test_split(subject_images, test_size=0.3, random_state=42)
            images.append((train_imgs, test_imgs))
            labels.append((np.full(len(train_imgs), label), np.full(len(test_imgs), label)))
            label += 1
    train_images = np.vstack([item[0] for item in images])
    test_images = np.vstack([item[1] for item in images])
    train_labels = np.concatenate([item[0] for item in labels])
    test_labels = np.concatenate([item[1] for item in labels])
    return train_images, train_labels, test_images, test_labels

def apply_pca(data_train, data_test, n_components):
    pca = PCA(n_components=n_components)
    data_train_pca = pca.fit_transform(data_train)
    data_test_pca = pca.transform(data_test)
    return data_train_pca, data_test_pca

def svm_classification(data_train, labels_train, data_test, labels_test, C_values):
    results = {}
    for C in C_values:
        svm = SVC(C=C, kernel='linear')
        svm.fit(data_train, labels_train)
        predictions = svm.predict(data_test)
        accuracy = accuracy_score(labels_test, predictions)
        results[C] = accuracy
    return results

def main():
    cmu_data_folder = 'CA2/PIE'  # Update with your actual path
    train_images, train_labels, test_images, test_labels = load_cmu_pie_data(cmu_data_folder)

    C_values = [0.01, 0.1, 1]
    pca_dimensions = [80, 200]
    results = {}

    print("SVM Classification using raw face images:")
    raw_results = svm_classification(train_images, train_labels, test_images, test_labels, C_values)
    for C, accuracy in raw_results.items():
        print(f"  C={C}: Accuracy = {accuracy:.4f}")
    results['raw'] = raw_results

    for n_components in pca_dimensions:
        print(f"\nSVM Classification using PCA with n_components={n_components}:")
        train_images_pca, test_images_pca = apply_pca(train_images, test_images, n_components)
        pca_results = svm_classification(train_images_pca, train_labels, test_images_pca, test_labels, C_values)
        for C, accuracy in pca_results.items():
            print(f"  C={C}: Accuracy = {accuracy:.4f}")
        results[f'pca_{n_components}'] = pca_results

    print("\nSummary of Results:")
    for key, res in results.items():
        print(f"\nResults for {key}:")
        for C, accuracy in res.items():
            print(f"  C={C}: Accuracy = {accuracy:.4f}")

if __name__ == "__main__":
    main()
