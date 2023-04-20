"""
    This file contains plot methods that show different heatmaps of the same image / model.
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from typing import List

from util.data_util import get_output_path


def plot_overview_for_question(
    human_heatmap: np.array,
    grad_cam_heatmap: np.array,
    att_rollout_heatmap: np.array,
    human_answers: List[str],
    model_answer: str,
    question: str,
    question_id: str
) -> None:
    """
        Method to plot different heatmap visualizations of the model in comparison to the human heatmap to the
        same image. The question as well as given answers are provided, as well. The resulting plot is shown
        and finally saved in an output folder of the current working directory.

        :param human_heatmap: The mean human heatmap
        :param grad_cam_heatmap: Model Grad-CAM heatmap
        :param att_rollout_heatmap: Model attention rollout heatmap
        :param human_answers: List of human answers
        :param model_answer: The given model answer
        :param question: The question as string
        :param question_id: The ID of the question
        :return: None
    """

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 8))

    # Construct lists to facilitate plotting
    titles = ["Mean human heatmap", "Model att-rollout heatmap", "Model Grad-CAM heatmap"]
    heatmaps = [human_heatmap, att_rollout_heatmap, grad_cam_heatmap]
    x_labels = [f"Human answers: {human_answers}", f"Model answer: {model_answer}", f"Model answer: {model_answer}"]

    # Plot to the corresponding axes
    for idx, ax in enumerate(axes):
        ax.set_title(titles[idx])
        ax.imshow(heatmaps[idx])
        ax.set_xlabel(x_labels[idx])

        # Turn off ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Some aesthetics stuff
    fig.suptitle(question)
    fig.tight_layout()
    fig.subplots_adjust(top=1.2)

    # Show and save plot
    plt.savefig(Path(get_output_path(), f"{question_id}_heatmap_comparison.png"))
    plt.show()
    plt.close()
