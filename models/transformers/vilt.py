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
    "ATTENTION_LAYER_HOOK_NAME": "attention.dropout",   # Name of the attention layers to consider
    "MODEL_WRAPPER_USED": False,    # Whether the model is wrapped in order to use a custom __call__ func for the hooks
    "UNSQUEEZE_ATTENTIONS": False,   # Whether the attentions should be unsqueezed after retrieval
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


def process_output(
    model: torch.nn.Module,
    processor: ViltProcessor,
    image_embeds: torch.FloatTensor,
    input_ids: torch.LongTensor,
    attention_mask=None,
    **kwargs
) -> str:
    """
        Method to process the output further. The logits are  decoded using the processor to get the final model answer
        token. The **kwargs are necessary to simply be able to plug all input and output into this function across all
        models, since not all models need the same input, but for consistency, all parameters need to remain.

        :param model: The model to use to further process the output, if necessary
        :param processor: The processor to use to decode logits into tokens
        :param image_embeds: The image embeddings from the model forward pass, if necessary (so far, only BLIP)
        :param input_ids: Input IDs for the question (encoded), if necessary
        :param attention_mask: Attention mask from the input, if necessary
        :param kwargs: To be able to simply pass all arguments from input and output, irrelevant ones are ignored
        :return: Final model answer token to the given question
    """

    # Index of the model answer token
    idx = kwargs["logits"].argmax(-1).item()

    # Decode using the token index and return
    return model.config.id2label[idx]
