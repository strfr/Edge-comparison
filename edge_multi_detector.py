import os
import cv2
import numpy as np

input_dir = "data"
output_base = "output_edges"
os.makedirs(output_base, exist_ok=True)

edge_detectors = ['Canny', 'Sobel', 'Laplacian', 'Scharr', 'Prewitt']
extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']

prewitt_kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

def apply_edge_detection(img_gray, method):
    if method == 'Canny':
        return cv2.Canny(img_gray, 100, 200)
    elif method == 'Sobel':
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif method == 'Laplacian':
        lap = cv2.Laplacian(img_gray, cv2.CV_64F)
        return cv2.convertScaleAbs(lap)
    elif method == 'Scharr':
        grad_x = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
        return cv2.convertScaleAbs(cv2.magnitude(grad_x, grad_y))
    elif method == 'Prewitt':
        grad_x = cv2.filter2D(img_gray, -1, prewitt_kx)
        grad_y = cv2.filter2D(img_gray, -1, prewitt_ky)
        return cv2.convertScaleAbs(cv2.addWeighted(grad_x, 0.5, grad_y, 0.5, 0))
    else:
        raise ValueError(f"Unsupported method: {method}")

for file in os.listdir(input_dir):
    if any(file.lower().endswith(ext) for ext in extensions):
        filepath = os.path.join(input_dir, file)
        img = cv2.imread(filepath)
        if img is None:
            print(f"Yüklenemedi: {filepath}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        filename_base = os.path.splitext(file)[0]

        for method in edge_detectors:
            edges = apply_edge_detection(gray, method)
            inverted = cv2.bitwise_not(edges) 

            method_dir = os.path.join(output_base, method)
            os.makedirs(method_dir, exist_ok=True)
            out_path = os.path.join(method_dir, f"{filename_base}_{method}.jpg")
            cv2.imwrite(out_path, inverted)
            print(f"{method} çıktısı: {out_path}")
