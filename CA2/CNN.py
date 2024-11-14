import numpy as np
import matplotlib.pyplot as plt
from dataset import load_data  # Ensure dataset.py is in the same directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

def preprocess_data(cmu_data_folder, selfie_folder, target_classes):
    train_images, train_labels, test_images, test_labels = load_data(cmu_data_folder, selfie_folder)
    
    train_filter = np.isin(train_labels, target_classes)
    test_filter = np.isin(test_labels, target_classes)
    
    train_images, train_labels = train_images[train_filter], train_labels[train_filter]
    test_images, test_labels = test_images[test_filter], test_labels[test_filter]
    
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(target_classes)}
    train_labels = np.array([label_mapping[label] for label in train_labels])
    test_labels = np.array([label_mapping[label] for label in test_labels])
    
    train_images = train_images.reshape(-1, 32, 32, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 32, 32, 1).astype('float32') / 255.0
    
    train_labels = to_categorical(train_labels, num_classes=len(target_classes))
    test_labels = to_categorical(test_labels, num_classes=len(target_classes))
    
    return train_images, train_labels, test_images, test_labels

def build_cnn(input_shape=(32, 32, 1), num_classes=26):
    model = Sequential()
    model.add(Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.show()

def evaluate_model(model, test_images, test_labels):
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    predictions = np.argmax(model.predict(test_images), axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    final_accuracy = accuracy_score(true_labels, predictions)
    print(f"Final Classification Accuracy: {final_accuracy:.4f}")

def main():
    np.random.seed(42)  # Set seed for reproducibility
    
    cmu_data_folder = 'CA2/PIE'  # Replace with your dataset path
    selfie_folder = 'CA2/PIE/Selfie'  # Update path
    target_classes = list(range(25)) + [69]
    
    train_images, train_labels, test_images, test_labels = preprocess_data(cmu_data_folder, selfie_folder, target_classes)
    
    print("Unique labels in training set:", np.unique(train_labels))
    print("Unique labels in test set:", np.unique(test_labels))
    
    model = build_cnn(num_classes=len(target_classes))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(test_images, test_labels))
    
    plot_training_history(history)
    evaluate_model(model, test_images, test_labels)

if __name__ == "__main__":
    main()
