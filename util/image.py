import cv2
import numpy as np

from PIL import Image
from typing import Dict, Tuple


def resize_array_to_img(img: Image.Image, array: np.array):
    """
        Resizes the given array to the given image.

        :param img: The image that the array is to be resized to.
        :param array: The array that is to be resized to the image's shape
        :return: Resized image
    """

    return cv2.resize(array, (img.size[0], img.size[1]))


def fuze_image_and_array(img: Image.Image, array: np.array, map_strength=0.8) -> np.array:
    """
        Fuzes the given array with the image by applying a colormap to the array,
        normalizing both image and array and displaying the array with a certain
        opacity.

        :param img: The image to be fuzed
        :param array: The array to be fuzed
        :param map_strength: The strength of the map, as float [0-1]
        :return: The fuzed image as an array
    """

    # Normalize image
    img = np.float32(img) / 255

    # Compute heatmap by applying a colormap and normalizing once more
    heatmap = cv2.applyColorMap(np.uint8(255 * array), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap * map_strength + np.float32(img)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)
