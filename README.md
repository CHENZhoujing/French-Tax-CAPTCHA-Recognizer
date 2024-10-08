# French Tax Authority CAPTCHA Recognition (For Learning Purposes)

This project is for **educational purposes only** and demonstrates how to recognize CAPTCHA images from the French tax authority.

## Project Overview

1. **Goal**:  
   Recognize and classify the digits in CAPTCHA images from the French tax authority.
   
2. **Approach**:  
   - **Data Collection**: Obtain around 20 CAPTCHA images.
   - **Image Processing**: Binarize each image to remove orange lines, isolating the digits.
   - **Digit Extraction**: Use K-means clustering to find the six digit locations in each CAPTCHA. Extract and save each digit as a 9x15 image.
   - **Model Training**:  
     Manually label the extracted digit images into categories (`0` to `9`). Train a Convolutional Neural Network (CNN) to recognize these digits.
   - **Results**:  
     The trained model can effectively recognize digits in new CAPTCHA images.

## Steps to Use the Project

1. **Preprocessing**:  
   Use the provided script to binarize CAPTCHA images and extract individual digits.

2. **Training**:  
   Train the CNN model with the labeled digit images. The model is tested to be functional.

3. **Prediction**:  
   Use the trained model to predict digits in new CAPTCHA images.

## Note

- This project is only intended for learning and exploring image recognition techniques.
- Do not use this code in violation of any website's terms of service.
