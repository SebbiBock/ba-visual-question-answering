"""
    File for the ViLT model, see:
    https://huggingface.co/dandelin/vilt-b32-finetuned-vqa
"""

from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from typing import Dict


CONFIG = {
    "MODEL_PATH": "dandelin/vilt-b32-finetuned-vqa",
    "PROCESSOR_PATH":  "dandelin/vilt-b32-finetuned-vqa",
    "PATCH_SIZE": 32,
    "ATTENTION_LAYER_HOOK_NAME": "attention.dropout"
}


def load_model():
    """
        Loads the model according to the given config, flags eval and returns it
    """

    model = ViltForQuestionAnswering.from_pretrained(CONFIG["MODEL_PATH"])
    model.eval()
    return model


def load_processor():
    """
        Loads the processor for the given model according to the config and returns it
    """

    return ViltProcessor.from_pretrained(CONFIG["PROCESSOR_PATH"])


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

    # Plug into processor and return
    return processor(image, question, return_tensors="pt")