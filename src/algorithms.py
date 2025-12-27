import cv2
import numpy as np
import time

class ImageProcessor:
    
    @staticmethod
    def add_noise(image, mode="s&p", amount=0.05):
        """Injects noise for benchmarking denoisers."""
        out = np.copy(image)
        if mode == "s&p":
            num_salt = np.ceil(amount * image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[tuple(coords)] = 255
            
            num_pepper = np.ceil(amount * image.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[tuple(coords)] = 0
        return out

    # --- Week 4: Interpolation ---
    @staticmethod
    def resize_image(image, scale, method):
        inter_map = {
            "Nearest Neighbor": cv2.INTER_NEAREST,
            "Linear": cv2.INTER_LINEAR,
            "Bicubic": cv2.INTER_CUBIC
        }
        start = time.time()
        res = cv2.resize(image, None, fx=scale, fy=scale, interpolation=inter_map[method])
        duration = (time.time() - start) * 1000 # ms
        return res, duration

    # --- Week 6: Grayscale & Thresholding ---
    @staticmethod
    def to_grayscale(image_rgb, method="Luma (BT-709)"):
        start = time.time()
        if method == "Average":
            res = np.mean(image_rgb, axis=2).astype(np.uint8)
        elif method == "Luma (BT-709)":
            res = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) 
        elif method == "Lightness":
            res = ((np.max(image_rgb, axis=2) + np.min(image_rgb, axis=2)) / 2).astype(np.uint8)
        else:
            res = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        duration = (time.time() - start) * 1000
        return res, duration

    @staticmethod
    def threshold(image_gray, method="Otsu", val=127):
        start = time.time()
        if method == "Otsu":
            _, res = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "Binary (Manual)":
            _, res = cv2.threshold(image_gray, val, 255, cv2.THRESH_BINARY)
        else:
            res = image_gray
        duration = (time.time() - start) * 1000
        return res, duration

    # --- Week 7: Filtering ---
    @staticmethod
    def apply_filter(image, filter_name, kernel_size=3):
        start = time.time()
        if filter_name == "Mean (Box)":
            res = cv2.blur(image, (kernel_size, kernel_size))
        elif filter_name == "Gaussian":
            res = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_name == "Median":
            res = cv2.medianBlur(image, kernel_size)
        elif filter_name == "Sobel Edge":
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
            res = cv2.magnitude(sobelx, sobely)
            res = np.uint8(res)
        elif filter_name == "Laplacian":
            res = cv2.Laplacian(image, cv2.CV_64F)
            res = np.uint8(np.absolute(res))
        else:
            res = image
        duration = (time.time() - start) * 1000
        return res, duration

    # --- Week 8: Morphology ---
    @staticmethod
    def apply_morphology(image, op_name, kernel_size=3):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        start = time.time()
        if op_name == "Erosion":
            res = cv2.erode(image, kernel, iterations=1)
        elif op_name == "Dilation":
            res = cv2.dilate(image, kernel, iterations=1)
        elif op_name == "Opening":
            res = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif op_name == "Closing":
            res = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif op_name == "Gradient":
            res = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        else:
            res = image
        duration = (time.time() - start) * 1000
        return res, duration