"""
    This file contains helper functions for data handling.
"""
import os

import pandas as pd

from pathlib import Path
from typing import List


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
