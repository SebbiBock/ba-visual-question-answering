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


def fuze_image_and_array(img: Image.Image, array: np.array) -> np.array:
    """
        Fuzes the given array with the image by applying a colormap to the array,
        normalizing both image and array and displaying the array with a certain
        opacity.

        :param img: The image to be fuzed
        :param array: The array to be fuzed:
        :return: The fuzed image as an array
    """

    # Normalize image
    img = np.float32(img) / 255

    # Compute heatmap by applying a colormap and normalizing once more
    heatmap = cv2.applyColorMap(np.uint8(255 * array), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def get_amount_of_image_patches(model_input: Dict, patch_size: int) -> Tuple[int, int]:
    """
        In Vision Transformers, the input image is divided into patches that have fixed sizes (the patch_size), and
        attention is not given for every pixel, but rather for each patch of size (patch_size x patch_size). Dependent
        on the input image, the amount of image patches might change if not all input images are transformed to the
        same size. Then, we need to get the amount of image patches that the image is divided in, e.g. necessary in
        determining the size of the heatmaps in attention rollout, since we have the attention given not for every
        pixel in the image, but rather for each pixel.

        :param model_input: The model input, containing the pixel_values for the preprocessed image
        :param patch_size: The patch size of the given model (one value, since patches are quadratic)
        :return: Tuple (amount_patches_h, amount_patches_w) containing the amount of patches in each dimension for the image.
    """

    # Compute the amount of image patches in width and height dimension: Divide image size by patch size
    amount_patches_w = int(model_input["pixel_values"].size(-1) / patch_size)
    amount_patches_h = int(model_input["pixel_values"].size(-2) / patch_size)

    return amount_patches_h, amount_patches_w
