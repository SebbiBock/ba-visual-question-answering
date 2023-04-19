"""
    File for the ViLT model, see:
    https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
"""

import torch

from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering, ViltConfig
from typing import Dict


CONFIG = {
    "MODEL_PATH": "dandelin/vilt-b32-finetuned-vqa",
    "PROCESSOR_PATH":  "dandelin/vilt-b32-finetuned-vqa",
    "PATCH_SIZE": 32,
    "ATTENTION_LAYER_HOOK_NAME": "attention.dropout",
    "LOGITS_OUTPUT_LEN": 3129   # From their paper: ViLT-VQA has 3129 answer classes
}


def load_model() -> ViltForQuestionAnswering:
    """
        Loads the model according to the given config, flags eval and returns it
    """

    model = ViltForQuestionAnswering.from_pretrained(CONFIG["MODEL_PATH"])
    model.eval()
    return model


def load_processor() -> ViltProcessor:
    """
        Loads the processor for the given model according to the config and returns it
    """

    return ViltProcessor.from_pretrained(CONFIG["PROCESSOR_PATH"])


def load_config() -> ViltConfig:
    """
        Loads the config for this specific pretrained model.
    """

    return ViltConfig.from_pretrained(CONFIG["MODEL_PATH"])


def preprocess(processor, question: str, answer: str, image: Image.Image) -> Dict:
    """
        For the given processor, image, question and answer, preprocess the data into the format used by the model.
        Also, implement additional preprocessing steps here.

        :param processor: The processor to use
        :param question: The question as a string
        :param answer: The annotated answer to the question as a string
        :param image: The image as a PIL.Image
        :return: The preprocessed data as a Dictionary.
    """

    # Plug image and question into processor
    inputs = processor(image, question, return_tensors="pt")

    # We need the label as one-hot vector, so get its proper token id from the config
    vilt_config = load_config()

    # Load its id from the config if the word exists, otherwise take closest one
    try:
        label_id = vilt_config.label2id[answer]
    except KeyError:

        # Take id if the token of the config is contained in the answer
        label_id = [token_id for token_id, token in vilt_config.id2label.items() if token in answer or answer in token][0]

    # Turn to onehot vector and append to output
    label = torch.zeros(CONFIG["LOGITS_OUTPUT_LEN"]).unsqueeze(0)
    label[0][label_id] = 1
    inputs["labels"] = label

    return inputs
