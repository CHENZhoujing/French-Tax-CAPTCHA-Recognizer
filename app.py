# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify, g
import cv2
import os
import numpy as np
import onnxruntime as ort
import uuid
import logging
import io

app = Flask(__name__)

# 设置日志
logging.basicConfig(level=logging.INFO)


# 加载预训练模型的函数，确保每个请求有自己的模型实例
def get_model():
    if 'model' not in g:
        g.model = ort.InferenceSession('digit_recognition_model.onnx')
    return g.model


def recognize_digits_from_image(img, model, gray_range=(45, 82)):
    """
    Recognize a sequence of 6 digits from the given image.

    Args:
        img (np.array): Image array.
        model: Pre-trained digit recognition model.
        gray_range (tuple): Gray level range for binarization.

    Returns:
        list: List of recognized digits.
    """
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
            model_input = digit_normalized.reshape(1, 10, 15, 1).astype(np.float32)

            # ONNX Runtime 推理
            ort_inputs = {model.get_inputs()[0].name: model_input}
            ort_outs = model.run(None, ort_inputs)
            prediction = ort_outs[0]
            predicted_label = np.argmax(prediction)
            recognized_digits.append(predicted_label)

    # Return the list of recognized digits
    return recognized_digits


@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传图片'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '未选择图片'}), 400

        # 使用内存中的文件
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': '无法加载图片'}), 400

        # 获取模型实例
        model = get_model()

        # 识别数字
        result = recognize_digits_from_image(img, model)

    except Exception as e:
        logging.error(f"Error during recognition: {e}")
        return jsonify({'error': '处理过程中出现错误'}), 500

    recognized_digits_str = ''.join(map(str, result))
    return jsonify({'recognized_digits': recognized_digits_str})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
