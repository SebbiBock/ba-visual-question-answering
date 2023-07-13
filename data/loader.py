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
from typing import Dict, List, Tuple, Union

import util.data_util as dutil


# Path to the data, if it differs from the current repo
DATA_PATH = "F:/Content/Bachelorarbeit/data/"

# Dictionary setting up all absolute paths
PATH_DICT = {
    "IMAGE_DIR_PATH": DATA_PATH + "vqav2/images/val2014/",
    "ANNOTATION_PATH": DATA_PATH + "vqav2/annotations/v2_mscoco_val2014_annotations.json",
    "ANNOTATION_TRAIN_PATH": DATA_PATH + "vqav2/annotations/v2_mscoco_train2014_annotations.json",
    "QUESTION_PATH": DATA_PATH + "vqav2/questions/v2_OpenEnded_mscoco_val2014_questions.json",
    "HUMAN_GAZE_PATH": DATA_PATH + "mhug/mhug/vqa-mhug_gaze.pickle",
    "HUMAN_ANSWERS_PATH": DATA_PATH + "mhug/mhug/vqa-mhug_answers.pickle",
    "BOUNDING_BOXES_PATH": DATA_PATH + "mhug/mhug/vqa-mhug_bboxes.pickle",
    "GENERATED_HEATMAP_DIR_PATH": DATA_PATH + "mhug/deliverables/vqa-mhug/img-attmap/",
    "EXPERIMENT_OUTPUT_PATH": "D:/6.Semester/Bachelorthesis/ba-visual-question-answering/experiment/exp_output/",
    "REASONING_TYPES_PATH": "D:/6.Semester/Bachelorthesis/ba-visual-question-answering/data/reasoning_types_saved.pkl",
    "EXPERIMENT_QUESTION_PATH": "D:/6.Semester/Bachelorthesis/ba-visual-question-answering/data/splits/"
}


def load_images(question_id_list: List[str], return_paths: bool = False) -> List[Union[Image.Image, Path]]:
    """
        Loads and returns the images for the given question_ids as a List.

        :param question_id_list: List of question ids (VQAv2), each as string.
        :param return_paths: Whether to return paths instead of loaded images. Defaults to False.
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
    return img_paths if return_paths else [Image.open(img_path) for img_path in img_paths]


def load_images_no_harddrive(question_id_list: List[str], return_paths: bool = False) -> List[Union[Image.Image, Path]]:
    """
        Loads and returns the images for the given question_ids as a List. Only works for the experiment images,
        can be used if I forget my harddrive again :)

        :param question_id_list: List of question ids (VQAv2), each as string.
        :param return_paths: Whether to return paths instead of loaded images. Defaults to False.
        :return: A List of loaded PIL.Images.
    """

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

    # Construct path to experiment data folders
    group_1_path = "D:/6.Semester/Bachelorthesis/ba-visual-question-answering/experiment/images/group1/"
    group_2_path = "D:/6.Semester/Bachelorthesis/ba-visual-question-answering/experiment/images/group2/"

    # Assign each q_id to either group1 or 2
    img_folder_paths = [group_1_path if os.path.isfile(os.path.join(group_1_path, q_id + ".jpg")) else group_2_path for
                        q_id in question_id_list]
    img_paths = [Path(q_ids_path + q_id + ".jpg") for q_ids_path, q_id in zip(img_folder_paths, question_id_list)]

    # Load and return images
    return img_paths if return_paths else [Image.open(img_path) for img_path in img_paths]


def load_questions(
        question_id_list: Union[List[str], None],
        fuze_ids_and_questions: bool = False
) -> Union[List[Tuple[str]], List[str], List[Dict[str, str]]]:
    """
        Loads and returns the questions for the given question_ids as a List.

        :param question_id_list: List of question ids (VQAv2), each as string or None, if all questions should be loaded
        :param fuze_ids_and_questions: Whether the questions are to be fuzed to their question ID.
        :return: A List of question strings, optionally with their IDs fuzed as Tuples
    """

    # Load in question annotation data
    with open(Path(PATH_DICT["QUESTION_PATH"]), "rb") as f:
        question_dict = json.load(f)

    if question_id_list is None:
        return question_dict["questions"]

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

    # Construct list
    question_list = [""] * len(question_id_list)

    # Iterate through all questions and get the proper question
    for q_dict in question_dict["questions"]:

        # If this question id is in the wanted list, place it at the proper index
        if str(q_dict["question_id"]) in question_id_list:
            question_list[question_id_list.index(str(q_dict["question_id"]))] = q_dict["question"]

    # Sanity check: No entry can be empty anymore
    assert "" not in question_list

    return list(zip(question_id_list, question_list)) if fuze_ids_and_questions else question_list


def load_annotated_answers(
        question_id_list: Union[List[str], None],
        single_answer: bool = True
) -> Dict[str, Union[str, List[str], Dict[str, List[str]]]]:
    """
        Loads and returns the annotated answers for the given question_ids. If the single_answer flag
        is set to true, only the most common answer across all annotators is returned, otherwise,
        all annotated answers are returned.

        :param question_id_list: List of question ids (VQAv2), each as string or None, if all answers are to be returned
        :param single_answer: If the most common annotated answer is to be returned, or all.
        :return: A Dictionary containing the question_ids as keys and the answer(s) as values, or the whole annotation json
    """

    # Load in annotated data from VQAv2
    with open(Path(PATH_DICT["ANNOTATION_PATH"]), "rb") as f:
        annotations = json.load(f)

    if question_id_list is None:
        return annotations["annotations"]

    # Sanity check: Enforce type
    question_id_list = [str(question_id) for question_id in question_id_list]

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
        Loads and returns all human answers.

        :return: pd.DataFrame containing all human answers.
    """

    # Load and reset multi-level index
    with open(Path(PATH_DICT["HUMAN_ANSWERS_PATH"]), "rb") as f:
        answer_df = pickle.load(f).reset_index()

    return answer_df


def load_participant_data(vp_code: Union[str, None] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Loads in all the data for the given participant and returns them in a Tuple. If no participant string
        is specified, the data for all participants is loaded, concatenated and returned.

        :param vp_code: The VP-Code if one specific participant is to be loaded, otherwise, all are loaded.
        :return: Tuple of (gaze_df, event_df, logger_df).
    """

    # Output directories to consider: ALl or the specified participant
    output_dirs = [x for x in os.listdir(Path(PATH_DICT["EXPERIMENT_OUTPUT_PATH"]))]
    if vp_code is not None:
        output_dirs = [x for x in output_dirs if x[-6:] == vp_code]

    # Loop through all given participant output directories and assemble list of dfs
    gaze_list = []
    event_list = []
    logger_list = []

    for part_dir in output_dirs:

        # Assemble paths to needed files
        path_to_gaze = os.path.join(PATH_DICT["EXPERIMENT_OUTPUT_PATH"], part_dir, "postprocessed", "gaze_df.pkl")
        path_to_events = os.path.join(PATH_DICT["EXPERIMENT_OUTPUT_PATH"], part_dir, "postprocessed", "event_df.pkl")
        path_to_logger = os.path.join(PATH_DICT["EXPERIMENT_OUTPUT_PATH"], part_dir, "postprocessed", "logger_df.pkl")

        # Read in data
        if os.path.isfile(path_to_gaze):
            with open(path_to_gaze, "rb") as fg:
                gaze_list.append(pickle.load(fg))

        if os.path.isfile(path_to_events):
            with open(path_to_events, "rb") as fe:
                event_list.append(pickle.load(fe))

        if os.path.isfile(path_to_logger):
            with open(path_to_logger, "rb") as fl:
                logger_list.append(pickle.load(fl))

        # Add VP Code identifier to every df
        for t_df in [gaze_list[-1], event_list[-1], logger_list[-1]]:
            t_df["vp_code"] = part_dir.split("-")[-1]

    # Some pre-processing for proper evaluation:
    for lgdf in logger_list:
        lgdf["answer"] = lgdf["answer"].apply(lambda x: "ANA" if x == "A N A " else x)

    # Concat the pd.DataFrames to one, if necessary, and return the tuple
    return (
        pd.concat(gaze_list, ignore_index=True) if len(gaze_list) > 1 else gaze_list[0],
        pd.concat(event_list, ignore_index=True) if len(event_list) > 1 else event_list[0],
        pd.concat(logger_list, ignore_index=True) if len(logger_list) > 1 else logger_list[0]
    )


def load_model_results(model_str: str) -> Tuple[Dict[str, np.array], Dict[str, np.array], pd.DataFrame]:
    """

    """

    # Get model path
    paths = dutil.create_model_output_folders(model_str, fail_on_non_existent=True)

    # Construct dicts for the heatmaps
    heatmap_types = ["att_rollout", "grad_cam"]
    return_dicts = [{}, {}]

    # Additional check for BEiT-3: No GRAD-CAM was computed here, since it is computationally expensive, and
    # we already opted to simply use the attention rollout maps
    if model_str == "beit3":
        heatmap_types = ["att_rollout"]

    for htype, return_dict in zip(heatmap_types, return_dicts):
        for heatmap_name in os.listdir(paths[htype]):
            return_dict.update({
                heatmap_name.split(".")[0]: np.load(os.path.join(paths[htype], heatmap_name))
            })

    # Load model answers
    with open(os.path.join(paths["predictions"], "answers.pkl"), "rb") as f:
        model_answer_df = pickle.load(f)

    return return_dicts[0], return_dicts[1], model_answer_df


def load_annotated_reasoning_types() -> Dict[str, str]:
    """
        Loads in the dictionary containing the reasoning type annotations for the question IDs of the VQAv2 dataset.

        :return: The dictionary containing question ids and their corresponding annotated reasoning type
    """

    with open(Path(PATH_DICT["REASONING_TYPES_PATH"]), "rb") as f:
        annotations = pickle.load(f)
    return annotations


def load_experiment_questions_for_group(group: int, only_values: bool = True) -> Union[List[int], pd.DataFrame]:
    """
        Loads in the questions actually used in the experiment for the given group. If only_values is True,
        only a one-dimensional array is returned containing the values, otherwise, the pd.DataFrame with
        reasoning type annotation as columns is returned.

        :param group: The group number
        :param only_values: Boolean flag whether to only retrieve the values (True) or the whole df (False)
        :return: Either the question ID values as a 1D List or the pd.DataFrame with reasoning type annotation
    """

    # Join path to point to proper group and load
    with open(os.path.join(PATH_DICT["EXPERIMENT_QUESTION_PATH"], f"used_questions_group_{group}.pkl"), "rb") as f:
        group_questions = pickle.load(f)

    if only_values:
        return group_questions.values.flatten().tolist()

    return group_questions


def load_demographic_data() -> pd.DataFrame:
    """
        Load and return the demographic data of all participants as one pd.DataFrame.

        :return: The pd.DataFrame containing all demographic data.
    """

    # Output directories to consider: ALl or the specified participant
    output_dirs = [x for x in os.listdir(Path(PATH_DICT["EXPERIMENT_OUTPUT_PATH"]))]

    # Return list
    demographic_lst = []

    for part_dir in output_dirs:

        # Assemble path to needed file
        path_to_demographic = os.path.join(PATH_DICT["EXPERIMENT_OUTPUT_PATH"], part_dir, "demographic.txt")

        # Read in data
        if os.path.isfile(path_to_demographic):
            with open(path_to_demographic, "r") as demog:
                lines = demog.readlines()
                demographic_lst.append([x.split(" ")[1].strip("\n") for x in lines[1:]])

    demog_df = pd.DataFrame(demographic_lst, columns=["Age", "Gender"])
    demog_df["Age"] = demog_df["Age"].astype(int)
    return demog_df