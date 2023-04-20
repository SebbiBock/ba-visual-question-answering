"""
    File for different functions that load in required data.
"""

import json
import pickle
import os

import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image
from typing import Dict, List, Union


# Path to the data, if it differs from the current repo
DATA_PATH = "F:/Content/Bachelorarbeit/data/"

# Dictionary setting up all absolute paths
PATH_DICT = {
    "IMAGE_DIR_PATH": DATA_PATH + "vqav2/images/val2014/",
    "ANNOTATION_PATH": DATA_PATH + "vqav2/annotations/v2_mscoco_val2014_annotations.json",
    "QUESTION_PATH": DATA_PATH + "vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json",
    "HUMAN_GAZE_PATH": DATA_PATH + "mhug/mhug/vqa-mhug_gaze.pickle",
    "HUMAN_ANSWERS_PATH": DATA_PATH + "",
    "BOUNDING_BOXES_PATH": DATA_PATH + "mhug/mhug/vqa-mhug_bboxes.pickle",
    "GENERATED_HEATMAP_DIR_PATH": DATA_PATH + "mhug/deliverables/vqa-mhug/img-attmap/"
}


def load_images(question_id_list: List[str]) -> List[Image.Image]:
    """
        Loads and returns the images for the given question_ids as a List.

        :param question_id_list: List of question ids (VQAv2), each as string.
        :return: A List of loaded PIL.Images.
    """

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

    # Get image ids from question ids
    image_id_list = [qid[:-3] for qid in question_id_list]

    # Construct VQAv2 file names
    img_filenames = [f"COCO_val2014_{'0' * (12 - len(image_id)) + image_id}.jpg" for image_id in image_id_list]
    img_paths = [Path(PATH_DICT["IMAGE_DIR_PATH"] + img_filename) for img_filename in img_filenames]

    # Load and return images
    return [Image.open(img_path) for img_path in img_paths]


def load_questions(question_id_list: List[str]) -> List[str]:
    """
        Loads and returns the questions for the given question_ids as a List.

        :param question_id_list: List of question ids (VQAv2), each as string.
        :return: A List of question strings.
    """

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

    # Load in question annotation data
    with open(Path(PATH_DICT["QUESTION_PATH"]), "rb") as f:
        question_dict = json.load(f)

    # Construct list
    question_list = [""] * len(question_id_list)

    # Iterate through all questions and get the proper question
    for q_dict in question_dict["questions"]:

        # If this question id is in the wanted list, place it at the proper index
        if str(q_dict["question_id"]) in question_id_list:
            question_list[question_id_list.index(str(q_dict["question_id"]))] = q_dict["question"]

    # Sanity check: No entry can be empty anymore
    assert "" not in question_list

    return question_list


def load_annotated_answers(question_id_list: List[str], single_answer: bool = True) -> Dict[str, Union[str, List[str]]]:
    """
        Loads and returns the annotated answers for the given question_ids. If the single_answer flag
        is set to true, only the most common answer across all annotators is returned, otherwise,
        all annotated answers are returned.

        :param question_id_list: List of question ids (VQAv2), each as string.
        :param single_answer: If the most common annotated answer is to be returned, or all.
        :return: A Dictionary containing the question_ids as keys and the answer(s) as values.
    """

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

    # Load in annotated data from VQAv2
    with open(Path(PATH_DICT["ANNOTATION_PATH"]), "rb") as f:
        annotations = json.load(f)

    # Construct dictionary
    answer_dict = {}

    # Get annotated answers
    for annotation_dict in annotations["annotations"]:
        if str(annotation_dict["question_id"]) in question_id_list:

            # Get all annotated answer for the given question
            answer = [ans["answer"] for ans in annotation_dict["answers"]]

            # Get most common answer if the flag is set
            if single_answer:
                answer = max(set(answer), key=answer.count)

            answer_dict.update({str(annotation_dict["question_id"]): answer})

    return answer_dict


def load_human_gaze() -> pd.DataFrame:
    """
        Loads and returns the entire human gaze dataframe.

        :return: pd.DataFrame containing all human gaze data
    """

    # Load in all human gaze data and reset the index
    with open(Path(PATH_DICT["HUMAN_GAZE_PATH"]), "rb") as f:
        gaze_df = pickle.load(f).reset_index()

    # Create image id column and return df
    gaze_df["image_id"] = gaze_df["question_id"].apply(lambda x: x[:-3])
    return gaze_df


def load_human_heatmaps(question_id: str, reduction: callable = None) -> Union[List[np.array], np.array]:
    """
        Loads the already generated heatmaps of every participant for the given question id. If a reduction
        callable is specified, it is used to reduce the heatmaps across all specified participants to one,
         otherwise, a List of heatmaps is returned.

        :param question_id: The question ID, as str
        :param reduction: Reductiom method to apply to all heatmaps (e.g. np.mean), defaults to None
        :return: Either a List of heatmaps or a reduced heatmap
    """

    # Load all heatmaps
    heatmap_paths = [x for x in os.listdir(Path(PATH_DICT["GENERATED_HEATMAP_DIR_PATH"])) if question_id in x]
    heatmaps = np.array([np.load(os.path.join(PATH_DICT["GENERATED_HEATMAP_DIR_PATH"], x)) for x in heatmap_paths])

    if reduction is not None:
        heatmaps = reduction(heatmaps, axis=0)

    return heatmaps


def load_bounding_boxes() -> pd.DataFrame:
    """
        Loads and returns the pd.DataFrame of the bounding boxes.

        :return: pd.DataFrame containing all bounding boxes
    """

    with open(Path(PATH_DICT["BOUNDING_BOXES_PATH"]), "rb") as f:
        boxes_df = pickle.load(f)

    return boxes_df


def load_human_answers() -> pd.DataFrame:
    """

    """

    pass

