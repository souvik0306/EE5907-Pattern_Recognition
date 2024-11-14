import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, img_size=(32, 32)):
    images = []
    labels = []
    label = 0
    for subject_folder in os.listdir(folder_path):
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

def load_data(cmu_data_folder, selfie_folder):
    images, labels = load_images_from_folder(cmu_data_folder)
    selfie_images = prepare_selfie_images(selfie_folder)

    # Split CMU PIE images into train and test sets for each subject
    unique_labels = np.unique(labels)
    train_images, train_labels, test_images, test_labels = [], [], [], []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        label_images = images[label_indices]
        label_train, label_test = train_test_split(label_images, test_size=0.3, random_state=42)
        train_images.extend(label_train)
        test_images.extend(label_test)
        train_labels.extend([label] * len(label_train))
        test_labels.extend([label] * len(label_test))

    train_images, train_labels = np.array(train_images), np.array(train_labels)
    test_images, test_labels = np.array(test_images), np.array(test_labels)

    # Prepare Selfie images (7 for training, 3 for testing)
    selfie_train, selfie_test = train_test_split(selfie_images, test_size=0.3, random_state=42)
    selfie_train_labels = np.full(len(selfie_train), label + 1)
    selfie_test_labels = np.full(len(selfie_test), label + 1)

    # Append selfie images to train and test sets
    train_images = np.vstack((train_images, selfie_train))
    train_labels = np.concatenate((train_labels, selfie_train_labels))
    test_images = np.vstack((test_images, selfie_test))
    test_labels = np.concatenate((test_labels, selfie_test_labels))

    return train_images, train_labels, test_images, test_labels
