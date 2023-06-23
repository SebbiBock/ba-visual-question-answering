import pandas as pd
import numpy as np

from typing import List, Tuple, Union


def create_gaussian_kernel_for_fixation(center, image_size, sig=(50, 50)):
    """
        Function that smooths over the given fixation using a gaussian kernel to create a heatmap of proper size
        (image size) for just that fixation.

        :param center: The fixation center as coordinate tuple (x,y).
        :param image_size: The total image size (width, height).
        :param sig: The sigma value for the smoothing of the gaussian kernel
        :return: Heatmap containing only this fixation smoothed with a gaussian kernel
    """

    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - center[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - center[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) / np.square(sig[0]) + np.square(yy) / np.square(sig[1])))
    return kernel


def create_gaussian_human_heatmap(
        fixation_df: pd.DataFrame,
        logger_df: pd.DataFrame,
        question_id: str,
        reduction: callable = None,
        scale_dur=True
) -> Union[List[np.array], np.array]:
    """
        Computes a gaussian human heatmaps from the given fixation data for the given question. The heatmaps are
        computed by applying a gaussian kernel over every given fixation and assembling the fixations to one heatmap
        that is then normalized. If a reduction callable is specified, it is used to reduce the heatmaps across all
        participants to one, otherwise, a List of heatmaps is returned, each entry containing one participant heatmap.

        :param fixation_df: The pd.DataFrame containing all valid fixation data
        :param logger_df: The pd.DataFrame containing all the bounding box data in coordinates
        :param question_id: The question ID, as str
        :param reduction: Reduction method to apply to all heatmaps (e.g. np.mean), defaults to None
        :param scale_dur: Whether the gaussian kernel strength should be scaled with the duration of a fixation
        :return: Either a List of heatmaps or a reduced heatmap
    """

    # Create return list
    gaussian_heatmap_list = []

    # Get bounding coordinates for the image
    logger_slice = logger_df[logger_df["question_id"] == question_id]
    x_min, x_max, y_min, y_max = logger_slice[["bb_image_x_min", "bb_image_x_max", "bb_image_y_min", "bb_image_y_max"]].values[0]

    # Get image dimensions
    img_w, img_h = int(x_max - x_min), int(y_max - y_min)

    # Get slice of the gaze dataframe corresponding to the given question id
    q_slice = fixation_df[fixation_df["question_id"] == question_id]

    # Check that the question id is actually given in the fixation df
    if q_slice.shape[0] == 0:
        print(f"WARNING:\tNo fixation data given for question_id {question_id}, returning an empty gaussian heatmap")
        return np.zeros((img_h, img_w))

    # Loop through every participant for the given question
    for p_id in q_slice["vp_code"].value_counts().index:

        # Create empty heatmap
        heatmap = np.zeros((img_h, img_w))

        # Get slice for this participant
        p_slice = q_slice[q_slice["vp_code"] == p_id]

        # Loop through every fixation along with its duration
        for fix_x, fix_y, fix_dur in p_slice[["image_x_start", "image_y_start", "duration"]].values:

            # Smooth over fixation with gaussian kernel
            fix_coords = (fix_x, fix_y)
            gauss_fix = create_gaussian_kernel_for_fixation(fix_coords, (img_w, img_h))

            # Scale with duration of fixation, if necessary
            if scale_dur:
                gauss_fix *= fix_dur

            # Add to heatmap
            heatmap += gauss_fix

        # Normalize
        heatmap = heatmap / np.max(heatmap)

        # Append this participant heatmap to list of all human heatmaps
        gaussian_heatmap_list.append(heatmap)

    # Apply reduction, if given
    if reduction is not None:
        gaussian_heatmap_list = reduction(gaussian_heatmap_list, axis=0)

    return gaussian_heatmap_list


def compute_low_res_human_heatmap(
        fixation_df: pd.DataFrame,
        logger_df: pd.DataFrame,
        question_id: str,
        att_map_shape: Tuple[int, int] = (24, 24),
        reduction: callable = None
) -> Union[List[np.array], np.array]:
    """
        Computes low-resolution human heatmaps from the given gaze data for the given question. The computed heatmaps
        match the provided low resolution size in order to have heatmaps in the same dimension as attention heatmaps
        of Transformers. For this, the method only considers fixations on the image plate and within the bounding
        boxes of the image. If a reduction callable is specified, it is used to reduce the heatmaps across all
        participants to one, otherwise, a List of heatmaps is returned.

        :param fixation_df: The pd.DataFrame containing all valid fixation data
        :param logger_df: The pd.DataFrame containing all the bounding box data in coordinates
        :param question_id: The question ID, as str
        :param att_map_shape: The shape of the low-res human heatmap that is to be computed. (24, 24) is for BLIP.
        :param reduction: Reduction method to apply to all heatmaps (e.g. np.mean), defaults to None
        :return: Either a List of heatmaps or a reduced heatmap
    """

    # Create empty return list
    lowres_human_heatmaps = []

    # Get bounding coordinates for the image
    logger_slice = logger_df[logger_df["question_id"] == question_id]
    x_min, x_max, y_min, y_max = logger_slice[["bb_image_x_min", "bb_image_x_max", "bb_image_y_min", "bb_image_y_max"]].values[0]

    # Get height and width of the low-res map of the model
    lr_h, lr_w = att_map_shape

    # Get slice of the gaze dataframe corresponding to the given question id
    q_slice = fixation_df[fixation_df["question_id"] == question_id]

    # Check that the question id is actually given in the fixation df
    if q_slice.shape[0] == 0:
        print(f"WARNING:\tNo fixation data given for question_id {question_id}, returning an empty low-res heatmap")
        return np.zeros(att_map_shape)

    # Create boundaries for the given image size according to the width and height of the attentions maps from the
    # Transformers. This defines ranges of pixels that correspond to one pixel in the low-res image
    lr_h_boundaries = np.linspace(start=y_min, stop=y_max, num=lr_h + 1)
    lr_w_boundaries = np.linspace(start=x_min, stop=x_max, num=lr_w + 1)

    # Loop through every participant for the given question
    for p_id in q_slice["vp_code"].value_counts().index:

        # Create empty low-res heatmap
        lowres_heatmap = np.zeros(att_map_shape)

        # Get slice for this participant
        p_slice = q_slice[q_slice["vp_code"] == p_id]

        # For every gaze coordinate point, update its proper cell in the heatmap
        for gaze_x, gaze_y in p_slice[["x_start", "y_start"]].values:

            # Get only valid fixations in the range of the image
            if x_min <= gaze_x <= x_max and y_min <= gaze_y <= y_max:
                # Get the proper index for the screen coordinates in the low-res heatmaps and insert them there
                # by finding the range in the boundaries that this pixel belongs to
                low_res_h_index = np.where(((lr_h_boundaries - int(gaze_y)) <= 0) == True)[0][-1]
                low_res_w_index = np.where(((lr_w_boundaries - int(gaze_x)) <= 0) == True)[0][-1]
                lowres_heatmap[low_res_h_index, low_res_w_index] += 1

        # Normalize the heatmap
        lowres_heatmap = lowres_heatmap / np.max(lowres_heatmap)

        # Append filled heatmap to heatmaps
        lowres_human_heatmaps.append(lowres_heatmap)

    # Apply reduction, if given
    if reduction is not None:
        lowres_human_heatmaps = reduction(lowres_human_heatmaps, axis=0)

    return lowres_human_heatmaps
