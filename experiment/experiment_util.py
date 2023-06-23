import os
import pandas as pd

from constants import *
from PIL import Image
from typing import Tuple



def get_participant_group() -> int:
    """
        Returns the group for the current participant based on the amount of generated files
        in the experiment output directory.

        :return: The group number, either 1 or 2.
    """

    # Get all the participants so far and use that to determine the group
    return (len(os.listdir(os.path.join(DIR, "exp_output"))) % 2) + 1


def get_group_image_directory(part_group: int):
    """
        Returns the directory where the experiment images are stored for the given participant group.
        If the directory does not yet exist, it will be created.

        :param part_group: The group of the participant
        :return: The path to the output directory
    """

    # Path to the image file for the given group
    img_dir = os.path.join(DIR, f'images/group{part_group}')

    # Create the directory, if it does not yet exist
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

        # Raise exception that no images could be found
        raise Exception(f"Aborting experiment: No image files found under {img_dir}!")

    return img_dir


def get_test_image_directory():
    """
        Returns the directory to the test images.

        :return: Directory to the test images.
    """

    # Path to the image file for the given group
    test_dir = os.path.join(DIR, f'images/test')

    # Create the directory, if it does not yet exist
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

        # Raise exception that no images could be found
        raise Exception(f"Aborting experiment: No image files found under {test_dir}!")

    return test_dir


def get_data_safety_dir():
    """
        Returns the directory to the pages of the data safety directory.

        :return: Directory to the data safety images.
    """

    # Path to the image file for the given group
    data_safety_dir = os.path.join(DIR, f'images/data_safety')

    # Create the directory, if it does not yet exist
    if not os.path.isdir(data_safety_dir):
        os.mkdir(data_safety_dir)

        # Raise exception that no images could be found
        raise Exception(f"Aborting experiment: No image files found under {data_safety_dir}!")

    return data_safety_dir


def create_and_return_output_directory(vp_name_str: str):
    """
        Creates and returns the output directory for the given participant.

        :param vp_name_str: The string representation for the current participant.
        :return: The output directory for the given participant.
    """

    # Directory for this participant where the data will be stored
    output_dir = os.path.join(DIR, f'exp_output/{vp_name_str}')

    # Create the directory, if it does not yet exist
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    return output_dir


def create_participant_string(vp_code: str) -> str:
    """
        Creates and returns the string representation for logging and saving purposes given the
        VP code of the participant and the current timestamp.

        :param vp_code: Code of the VP
        :return: Participant string
    """

    # Get current timestamp and convert to usable string for name of the logging file
    timestamp_str = pd.Timestamp.now().strftime('%Y-%m-%d-%H-%M')

    # Append with the vp code and return
    return timestamp_str + "-" + vp_code


def save_demographic_data(
    output_dir: str,
    age: str,
    gender: str,
    group: int
):
    """
        Saves the demographic data for the given participant to their output directory.
    """

    # Create file for the given participant in their directory
    f = open(os.path.join(output_dir, "demographic.txt"), "x")

    # Write to it and then close it
    f.write(f"Group: {group}\n")
    f.write(f"Age: {age}\n")
    f.write(f"Gender: {gender}\n")
    f.close()


def get_image_scale(img_path: str, disp_size: Tuple[int], factor: Tuple[float, float] = (0.75, 0.85)) -> Tuple[int]:
    """
        Returns the scale factor for the given image so that its longer side (either width or height, but prefer
        height over width in equal cases) is as long as a certain percentage / factor of the display size.

        :param img_path: Path to the image that is to be scaled
        :param disp_size: The current display size
        :param factor: The factors by which to scale the image. Separated into dimensions.
    """

    # Open the image and get the size
    img_size = Image.open(img_path).size

    img_w, img_h = img_size
    longer_idx = 0 if img_w > img_h else 1
    shorter_idx = 1 if img_w > img_h else 0

    # Calculate scale factor so that the longer side is as long as the given percentage of the display size
    # Example: img_size = (640, 480), disp_size = (1920, 1080), factor = 0.75
    # Example: -> Desired size = 1440 -> scale factor = 1440 / 640 = 2.25
    desired_size = int(disp_size[longer_idx] * factor[longer_idx])

    # Calculate scale factor
    scale_factor = desired_size / img_size[longer_idx]

    # Check the smaller side: If the smaller side would be out of bounds with safety margin, use the factor for
    # the smaller size instead. This only happens for near-quadratic images where the width is slightly longer.
    if img_size[shorter_idx] * scale_factor >= disp_size[shorter_idx] * 0.95:
        print("Hey")
        return int(disp_size[shorter_idx] * factor[shorter_idx]) / img_size[shorter_idx]

    return scale_factor



