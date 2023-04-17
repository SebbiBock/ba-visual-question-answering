"""
    File for the Blip model, see:
    https://huggingface.co/Salesforce/blip-vqa-base
"""

from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Dict


CONFIG = {
    "MODEL_PATH": "Salesforce/blip-vqa-base",
    "PROCESSOR_PATH":  "Salesforce/blip-vqa-base",
    "PATCH_SIZE": 16,
    "ATTENTION_LAYER_HOOK_NAME": "self_attn.dropout"
}


def load_model():
    """
        Loads the model according to the given config, flags eval and returns it
    """

    model = BlipForQuestionAnswering.from_pretrained(CONFIG["MODEL_PATH"])
    model.eval()
    return model


def load_processor():
    """
        Loads the processor for the given model according to the config and returns it
    """

    return BlipProcessor.from_pretrained(CONFIG["PROCESSOR_PATH"])


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

    # Plug into processor
    inputs = processor(image.convert('RGB'), question, return_tensors="pt")

    # Process the answer, get the input ids and append to the input dict
    label = processor(text=answer, return_tensors="pt").input_ids
    inputs["labels"] = label

    return inputs
