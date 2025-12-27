# Image Processing Benchmark & Hybridization Suite

A comprehensive Computer Vision benchmarking tool built with Python and Streamlit. This project allows users to compare various image processing algorithms  and create hybrid combinations of algorithms.

## 🚀 Features

### 1. Benchmark Suite
Compare algorithms based on **Execution Time (ms)** and **Quality (Error Metrics)**.
- **Interpolation:** Nearest Neighbor vs. Bilinear vs. Bicubic.
- **Noise Removal:** Mean vs. Median filters on Salt & Pepper noise.
- **Thresholding:** Global Manual vs. Otsu's Automatic Method.
- **Grayscale:** Average vs. Luma (BT-709) vs. Lightness.

### 2. Hybrid Merge Studio
Combine two different algorithms to create a composite result.
- **Inputs:** Select two algorithms (e.g., Gaussian Blur + Sobel Edge).
- **Logic:** Merge them using Weighted Average, Max, Min, or Difference.
- **Application:** Create robust edge detectors or band-pass filters.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/image-processing-benchmark.git](https://github.com/YOUR_USERNAME/image-processing-benchmark.git)
   cd image-processing-benchmark


### how to start ##
#first#
pip install -r requirements.txt 
#then#
streamlit run app.py
