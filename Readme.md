# Celebrity Recognition with Image Processing and Wavelet Transforms

This project implements a **celebrity recognition system** using machine learning techniques. The project involves processing images of various celebrities, applying a **wavelet transform** to extract texture features, and combining them with raw pixel data to create feature vectors. These vectors are then used to train a machine learning model to classify the celebrity in each image.

## Project Overview
The project uses **wavelet transforms** (specifically the **Daubechies wavelet**) to extract high-frequency texture features from celebrity images. These features are then combined with raw pixel data (resized to 32x32 pixels) to form a feature vector. The model is trained on these vectors for **celebrity classification**.

### Key Steps:
1. **Data Preprocessing**: Images are read and resized to 32x32 pixels.
2. **Wavelet Transform**: The raw images are transformed using the Daubechies wavelet (`db1`), which highlights finer details like edges and textures.
3. **Feature Extraction**: The raw and wavelet-transformed images are flattened into 1D vectors and combined into a single feature vector.
4. **Model Training**: A machine learning model is trained on the extracted features to classify images based on the celebrity's name.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `opencv-python` (cv2)
- `pywt` (PyWavelets)
- `sklearn` (for machine learning models)


