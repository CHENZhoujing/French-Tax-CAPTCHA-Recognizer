# digit_recognition_training.py

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


def load_digit_images(data_dir):
    """
    Load digit images and labels from the specified directory.

    Parameters:
    - data_dir: str, path to the dataset directory where subdirectories are named with digit labels (0-9).

    Returns:
    - images: numpy.ndarray, array of image data.
    - labels: numpy.ndarray, array of corresponding labels.
    """
    images = []
    labels = []
    # Get list of digit folders (0-9)
    for label in sorted(os.listdir(data_dir)):
        label_path = os.path.join(data_dir, label)
        if os.path.isdir(label_path) and label.isdigit():
            # Iterate over image files in each digit folder
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                try:
                    # Open the image
                    img = Image.open(img_path)
                    # Ensure image size is (10, 15), resize if necessary (width, height)
                    img = img.resize((10, 15))
                    # Convert image to grayscale
                    img = img.convert('L')
                    img_array = np.array(img)
                    # Normalize pixel values to [0, 1]
                    img_array = img_array / 255.0
                    images.append(img_array)
                    labels.append(int(label))
                except Exception as e:
                    print(f"Cannot load image {img_path}, error: {e}")
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def main():
    # 1. Specify the dataset path
    data_dir = '/digital_dataset'  # Replace with your dataset path

    # 2. Load data
    images, labels = load_digit_images(data_dir)

    # 3. Adjust data shape
    # Add channel dimension, shape becomes (num_samples, height, width, channels)
    images = images.reshape(-1, 15, 10, 1)

    # 4. Split the dataset
    # Split data into training set and test set (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # 5. Build the model
    model = Sequential([
        Conv2D(25, (4, 4), activation='relu', input_shape=(15, 10, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Assuming 10 classes (digits 0-9)
    ])

    # 6. Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 7. Train the model
    epochs = 300  # Adjust as needed
    batch_size = 2  # Adjust based on dataset size

    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,  # Use 20% of training data as validation set
                        verbose=1)

    # 8. Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    # 9. Save the model
    model.save('digit_recognition_model.h5')

    # 10. Visualize training results
    # Plot accuracy curve
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
