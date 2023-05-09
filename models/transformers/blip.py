"""
    File for the Blip model, see:
    https://huggingface.co/Salesforce/blip-vqa-base
"""

import torch

from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Dict, Tuple, Union


CONFIG = {
    "MODEL_PATH": "Salesforce/blip-vqa-base",
    "PROCESSOR_PATH":  "Salesforce/blip-vqa-base",
    "PATCH_SIZE": 16,
    "ATTENTION_LAYER_HOOK_NAME": "self_attn.dropout",   # Name of the attention layers to consider
    "GRAD_LAYERS_HOOK_LIST": "[model.vision_model.encoder.layers[-2].layer_norm2]",  # List of layer names where the hooks for gradient and activation saving are to be saved
    "MODEL_WRAPPER_USED": False,    # Whether the model is wrapped in order to use a custom __call__ func for the hooks
    "UNSQUEEZE_ATTENTIONS": False,   # Whether the attentions should be unsqueezed after retrieval
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


def process_output(
    model: torch.nn.Module,
    processor: BlipProcessor,
    image_embeds: torch.FloatTensor,
    input_ids: torch.LongTensor,
    attention_mask=None,
    **kwargs
) -> str:
    """
        Method to process the output further. Since the forward method of BLIP is usually only used for training,
        it does not return the output token. To fix this, the additional steps in generation for the BLIP model
        are executed with given model, input and image_embeds (from the output). The final logits result is then
        decoded using the processor to get the final model answer token. The **kwargs are necessary to simply be
        able to plug all input and output into this function across all models.

        :param model: The model to use to further process the output, if necessary
        :param processor: The processor to use to decode logits into tokens
        :param image_embeds: The image embeddings from the model forward pass, if necessary (so far, only BLIP)
        :param input_ids: Input IDs for the question (encoded), if necessary
        :param attention_mask: Attention mask from the input, if necessary
        :param kwargs: To be able to simply pass all arguments from input and output, irrelevant ones are ignored
        :return: Final model answer token to the given question
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

        # Create proper logits output
        logits_tensor = model.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=model.config.text_config.sep_token_id,
            pad_token_id=model.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=question_attention_mask
        )

        # Decode the given logits tensor and return the result
        return processor.decode(logits_tensor[0], skip_special_tokens=True)


def get_textual_embedding_length(model_input: Dict) -> int:
    """
        This function calculates the textual embedding length for the given question. This means that the amount
        of token (ids) that are passed into the Embedder of the Transformer are determined. If no attention is
        provided on the questions, simply return 0.

        :param model_input: The model input for the given question.
        :return: The model input token amount, or 0 if no attention is used on question tokens.
    """

    return 0


def image_patch_embedding_retrieval_fct(a: torch.Tensor, text_embed_length: int) -> Union[torch.Tensor, None]:
    """
        Callable function to reduce the attention matrix to only attention on the image embeddings, if attention is
        deployed on the textual input. Since the order of visual and textual input is dependent on the model, this is
        outsourced. If no attention is given on the text, simply return None.

        :param a: The attention matrix for one attention block.
        :param text_embed_length: The length of the tokens of the input question.
    """

    return None


def get_amount_of_image_patches(model_input: Dict) -> Tuple[int, int]:
    """
        In Vision Transformers, the input image is divided into patches that have fixed sizes (the patch_size), and
        attention is not given for every pixel, but rather for each patch of size (patch_size x patch_size). Dependent
        on the input image, the amount of image patches might change if not all input images are transformed to the
        same size. Then, we need to get the amount of image patches that the image is divided in, e.g. necessary in
        determining the size of the heatmaps in attention rollout, since we have the attention given not for every
        pixel in the image, but rather for each pixel.

        :param model_input: The model input, containing the pixel_values for the preprocessed image
        :return: Tuple (amount_patches_h, amount_patches_w) containing the amount of patches in each dimension for the image.
    """

    # The patch size of the given model (one value, since patches are quadratic)
    patch_size = CONFIG["PATCH_SIZE"]

    # Compute the amount of image patches in width and height dimension: Divide image size by patch size
    amount_patches_w = int(model_input["pixel_values"].size(-1) / patch_size)
    amount_patches_h = int(model_input["pixel_values"].size(-2) / patch_size)

    return amount_patches_h, amount_patches_w
