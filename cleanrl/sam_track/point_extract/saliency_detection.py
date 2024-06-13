img_d = f"/home/{USER}/Downloads/Segment-and-Track-Anything/point_extract/input_image.jpg"
import cv2
import numpy as np
import matplotlib.pyplot as plt

def saliency_detection(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

    success, saliency_map = saliency.computeSaliency(image_gray)

    saliency_map = (saliency_map * 255).astype("uint8")

    return saliency_map

def binarize_saliency_map(saliency_map, threshold=128):
    _, binary_map = cv2.threshold(saliency_map, threshold, 255, cv2.THRESH_BINARY)
    return binary_map

def remove_noise(binary_map, kernel_size=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary_map_cleaned = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary_map_cleaned

def plot_gray_image(gray_image, title="", cmap="gray"):
    plt.imshow(gray_image, cmap=cmap)
    plt.axis("off")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    image_path = img_d  # 请替换为您的图像路径
    gray_saliency_map = saliency_detection(image_path)
    plot_gray_image(gray_saliency_map, title="Saliency Detection Result")

    binary_saliency_map = binarize_saliency_map(gray_saliency_map)
    plot_gray_image(binary_saliency_map, title="Binary Saliency Map", cmap="gray")

    noise_removed_saliency_map = remove_noise(binary_saliency_map)
    plot_gray_image(noise_removed_saliency_map, title="Noise Removed Binary Saliency Map", cmap="gray")