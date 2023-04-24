import pandas as pd
import numpy as np

from typing import List, Tuple, Union


def compute_human_heatmap(
    gaze_df: pd.DataFrame,
    bboxes_df: pd.DataFrame,
    question_id: str,
    att_map_shape: Tuple[int] = (24, 24),
    reduction: callable = None
) -> Union[List[np.array], np.array]:
    """
        Computes low-resolution human heatmaps from the given gaze data for the given question. The computed heatmaps
        match the provided low resolution size in order to have heatmaps in the same dimension as attention heatmaps
        of Transformers. For this, the method only considers fixations on the image plate and within the bounding
        boxes of the image. If a reduction callable is specified, it is used to reduce the heatmaps across all
        participants to one, otherwise, a List of heatmaps is returned.

        :param gaze_df: The pd.DataFrame containing all gaze data
        :param bboxes_df: The pd.DataFrame containing all the bounding box data in coordinates
        :param question_id: The question ID, as str
        :param att_map_shape: The shape of the low-res human heatmap that is to be computed. (24, 24) is for BLIP.
        :param reduction: Reduction method to apply to all heatmaps (e.g. np.mean), defaults to None
        :return: Either a List of heatmaps or a reduced heatmap
    """

    # Create empty return list
    lowres_human_heatmaps = []

    # Get slice of the gaze dataframe corresponding to the given question id with fixations only on the image
    q_slice = gaze_df[(gaze_df["question_id"] == question_id) & (gaze_df["plate"] == "imgplate")]

    # Get bounding coordinates for the image
    q_id_bboxes = bboxes_df.loc[question_id]
    y_min, x_min, y_max, x_max = q_id_bboxes[q_id_bboxes["token"] == "IMG"][["ymin", "xmin", "ymax", "xmax"]].values[0]

    # Get height and width of the low-res map of the model
    lr_h, lr_w = att_map_shape

    # Create boundaries for the given image size according to the width and height of the attentions maps from the
    # Transformers. This defines ranges of pixels that correspond to one pixel in the low-res image
    lr_h_boundaries = np.linspace(start=y_min, stop=y_max, num=lr_h + 1)
    lr_w_boundaries = np.linspace(start=x_min, stop=x_max, num=lr_w + 1)

    # Loop through every participant for the given question
    for p_id in q_slice["participant_id"].value_counts().index:

        # Create empty low-res heatmap
        lowres_heatmap = np.zeros(att_map_shape)

        # Get slice for this participant
        p_slice = q_slice[q_slice["participant_id"] == p_id]

        # For every gaze coordinate point, update its proper cell in the heatmap
        for gaze_x, gaze_y in p_slice[["x", "y"]].values:

            # Get only valid fixations in the range of the image
            if x_min <= gaze_x <= x_max and y_min <= gaze_y <= y_max:

                # Get the proper index for the screen coordinates in the low-res heatmaps and insert them there
                # by finding the range in the boundaries that this pixel belongs to
                low_res_h_index = np.where(((lr_h_boundaries - int(gaze_y)) <= 0) == True)[0][-1]
                low_res_w_index = np.where(((lr_w_boundaries - int(gaze_x)) <= 0) == True)[0][-1]
                lowres_heatmap[low_res_h_index, low_res_w_index] += 1

        # Append filled heatmap to heatmaps
        lowres_human_heatmaps.append(lowres_heatmap)

    # Apply reduction, if given
    if reduction is not None:
        lowres_human_heatmaps = reduction(lowres_human_heatmaps, axis=0)

    return lowres_human_heatmaps
