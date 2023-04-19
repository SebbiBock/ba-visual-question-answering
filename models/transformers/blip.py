"""
    File for the Blip model, see:
    https://huggingface.co/Salesforce/blip-vqa-base
"""

import torch

from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Dict, Optional


CONFIG = {
    "MODEL_PATH": "Salesforce/blip-vqa-base",
    "PROCESSOR_PATH":  "Salesforce/blip-vqa-base",
    "PATCH_SIZE": 16,
    "ATTENTION_LAYER_HOOK_NAME": "self_attn.dropout"
}


def load_model() -> BlipForQuestionAnswering:
    """
        Loads the model according to the given config, flags eval and returns it
    """

    model = BlipForQuestionAnswering.from_pretrained(CONFIG["MODEL_PATH"])
    model.eval()
    return model


def load_processor() -> BlipProcessor:
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


def process_output(model: torch.nn.Module, image_embeds: torch.FloatTensor, input_ids: torch.LongTensor, attention_mask=None) -> torch.LongTensor:
    """
        Method to process the output further. Since the forward method of BLIP is usually only used for training,
        it does not return the output token. To fix this, the additional steps in generation for the BLIP model
        are executed with given model, input and image_embeds (from the output).
    """

    with torch.no_grad():

        # Compute image attention mask from image embeds
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        # Get question output to get question embedding
        question_embeds = model.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=False,
        )[0]

        # Compute question attention mask
        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long)

        bos_ids = torch.full(
            (question_embeds.size(0), 1), fill_value=model.decoder_start_token_id
        )

        # Create proper logits output and return it
        return model.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=model.config.text_config.sep_token_id,
            pad_token_id=model.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask
        )