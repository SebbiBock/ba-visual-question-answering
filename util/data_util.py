"""
    This file contains helper functions for data handling.
"""

import pickle
import os

import pandas as pd
import numpy as np

from pathlib import Path
from typing import Dict, List


def get_answers_for_question(answer_df: pd.DataFrame, question_id: str) -> List[str]:
    """
        Returns a List of human answers for the given question_id.

        :param answer_df: pd.DataFrame with single-level index containing all answers
        :param question_id: The question ID to consider, as string
        :return: List of all human answers
    """

    return answer_df[answer_df["question_id"] == question_id]["answer"].values.tolist()


def construct_output_folder() -> None:
    """
        An output folder is created in the current cwd, if the folder does not already exist.
    """

    if not os.path.isdir(get_output_path()):
        os.mkdir(get_output_path())


def get_output_path() -> Path:
    """
        Return the path to the output folder.
    """

    return Path(os.path.join(os.getcwd(), "outputs"))


def create_model_output_folders(model_str: str) -> Dict[str, str]:
    """
        Creates output directories for the given model and return the paths as a dictionary.

        :param model_str: String identifier for the given model.
        :return: Dictionary with the given output paths
    """

    # Construct path dict
    path_dict_model = {}

    # General model output path
    general_model_path = os.path.join(get_output_path().__str__(), "models")
    if not os.path.isdir(general_model_path):
        os.mkdir(general_model_path)

    # Specific model output path
    model_output_path = os.path.join(general_model_path, model_str)
    if not os.path.isdir(model_output_path):
        os.mkdir(model_output_path)

    # Construct output folders
    model_output_dirs = ["att_rollout", "grad_cam", "predictions"]
    for output_dir in model_output_dirs:
        t_path = os.path.join(model_output_path, output_dir)
        path_dict_model[output_dir] = t_path
        if not os.path.isdir(t_path):
            os.mkdir(t_path)

    return path_dict_model


def save_att_heatmaps(model_output_paths: Dict[str, str], question_ids: List[str], att_heatmaps: List[np.array]):
    """
        Save the attention heatmaps for the given model under the given path with the respective question_ids as
        name.

        :param model_output_paths: The dictionary containing all the model output paths
        :param question_ids: List of question IDs
        :param att_heatmaps: List of attention heatmaps
        :return: None
    """

    for q_id, heatmap in zip(question_ids, att_heatmaps):
        output_path = os.path.join(model_output_paths["att_rollout"], q_id)
        np.save(output_path, heatmap)


def save_grad_heatmaps(model_output_paths: Dict[str, str], question_ids: List[str], grad_cam_heatmaps: List[np.array]):
    """
        Save the GRAD-CAM heatmaps for the given model under the given path with the respective question_ids as
        name.

        :param model_output_paths: The dictionary containing all the model output paths
        :param question_ids: List of question IDs
        :param grad_cam_heatmaps: List of GRAD-CAM heatmaps
        :return: None
    """

    for q_id, heatmap in zip(question_ids, grad_cam_heatmaps):
        output_path = os.path.join(model_output_paths["grad_cam"], q_id)
        np.save(output_path, heatmap)


def save_predictions(model_output_paths: Dict[str, str], question_ids: List[str], model_answers: List[str]):
    """
        Save the predictions for the given model under the given path with the respective question_ids as
        name.

        :param model_output_paths: The dictionary containing all the model output paths
        :param question_ids: List of question IDs
        :param model_answers: List of model answers
        :return: None
    """

    # Output path to the prediction dataframe
    output_path = os.path.join(model_output_paths["predictions"], "answers.pkl")

    # If it already exists: Read it in
    if os.path.isfile(output_path):
        with open(output_path, "rb") as f:
            df = pickle.load(f)

    # Otherwise, create a new dataframe
    else:
        df = pd.DataFrame(columns=["question_id", "answer"])

    # Extend the dataframe
    for q_id, ans in zip(question_ids, model_answers):
        df.loc[df.shape[0]] = [q_id, ans]

    # Dump the new dataframe
    with open(output_path, "wb") as f:
        pickle.dump(df, f)


