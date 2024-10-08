# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model


def recognize_digits(image_path, model, gray_range=(45, 82)):
    """
    Recognize a sequence of 6 digits from the given image.

    Args:
        image_path (str): Path to the image file.
        model: Pre-trained digit recognition model.
        gray_range (tuple): Gray level range for binarization.

    Returns:
        list: List of recognized digits.
    """
    # Normalize the image path
    normalized_path = os.path.normpath(image_path)

    # Read the image
    img = cv2.imread(normalized_path)

    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return []

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binarize the image based on gray level range
    binary_img = np.full_like(gray_img, 255)  # Initialize white image
    binary_img[(gray_img >= gray_range[0]) & (gray_img <= gray_range[1])] = 0  # Set target gray levels to black

    # Find coordinates of black pixels
    black_pixel_coords = np.column_stack(np.where(binary_img == 0))

    # Apply K-means clustering to segment digits
    k = 6  # Number of clusters/digits
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.001)
    _, labels, centers = cv2.kmeans(
        black_pixel_coords.astype(np.float32),
        k,
        None,
        criteria,
        50,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers to integer coordinates
    centers = centers.astype(int)

    # Sort clusters from left to right based on x-coordinate
    sorted_cluster_indices = np.argsort(centers[:, 1])

    # Initialize list to store recognized digits
    recognized_digits = []

    # Process each cluster to extract and recognize digits
    for idx in sorted_cluster_indices:
        # Create mask for current cluster
        mask = np.zeros_like(binary_img)
        cluster_pixels = black_pixel_coords[labels.flatten() == idx]
        mask[cluster_pixels[:, 0], cluster_pixels[:, 1]] = 255

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            # Compute minimum area rectangle for the contour
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            # Get width and height of the rectangle
            width, height = int(rect[1][0]), int(rect[1][1])

            # Adjust width and height to standard size
            if width > height:
                width, height = 15, 10
            else:
                width, height = 10, 15

            # Compute perspective transform matrix
            src_pts = box.astype("float32")
            dst_pts = np.array(
                [
                    [0, height - 1],
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1]
                ],
                dtype="float32"
            )
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # Apply perspective transformation to get the digit image
            digit_image = cv2.warpPerspective(mask, M, (width, height))

            # Rotate image if necessary
            if width > height:
                digit_image = cv2.rotate(digit_image, cv2.ROTATE_90_CLOCKWISE)

            # Resize digit image to model input size
            digit_resized = cv2.resize(digit_image, (10, 15))  # (width, height)

            # Normalize pixel values
            digit_normalized = digit_resized / 255.0

            # Reshape for model input
            model_input = digit_normalized.reshape(1, 10, 15, 1)

            # Predict digit using the model
            prediction = model.predict(model_input)
            predicted_label = np.argmax(prediction)
            recognized_digits.append(predicted_label)

    # Return the list of recognized digits
    return recognized_digits


if __name__ == '__main__':
    # Load the pre-trained model
    model = load_model('digit_recognition_model.h5')

    # Specify the image path
    image_path = 'testdata/téléchargement (24).png'  # Replace with your image path

    # Recognize digits in the image
    result = recognize_digits(image_path, model)
    print("Recognized digits:", result)
