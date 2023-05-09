# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Adapted, Refactored and Expanded for own use as part of my Bachelor's Thesis
# --------------------------------------------------------'


import os
import torch
from timm.models import create_model
from pathlib import Path
from typing import Dict, List, Tuple, Union

import data.loader as loader

from models.transformers.beit3.engine_for_finetuning import get_handler, evaluate
from models.transformers.beit3.datasets import create_downstream_dataset
import models.transformers.beit3.utils as utils

# Leave this import here even though it is flagged as unused!
import models.transformers.beit3.modeling_finetune


FILE_PATH = Path(os.path.abspath(__file__)).parent.__str__()


CONFIG = {
    "MODEL": "beit3_large_patch16_480",
    "INPUT_SIZE": 480,
    "BATCH_SIZE": 16,
    "SENTENCEPIECE_MODEL": Path(FILE_PATH + r"\pretrained\beit3.spm").__str__(),
    "FINETUNE": Path(FILE_PATH + r"\pretrained\beit3_large_patch16_480_vqa.pth").__str__(),
    "EVAL": True,
    "MODEL_KEY": "model|module",
    "DROP_PATH":0.1,
    "MODEL_PREFIX": "",
    "VOCAB_SIZE": 64010,
    "NUM_MAX_BPE_TOKENS": 64,
    "EVAL_BATCH_SIZE": None,

    "ATTENTION_LAYER_HOOK_NAME": "self_attn.dropout_module",    # Name of the attention layers to consider
    "GRAD_LAYERS_HOOK_LIST": "[model.vilt.encoder.layer[-2].layernorm_before]", # List of layer names where the hooks for gradient and activation saving are to be saved
    "MODEL_WRAPPER_USED": True,  # Whether the model is wrapped in order to use a custom __call__ func for the hooks
    "UNSQUEEZE_ATTENTIONS": True    # Whether the attentions should be unsqueezed after retrieval
}


def load_model():
    """
        Loads the model according to the given config, flags eval and returns it
    """

    return BEiT3WrapperForEncapsulators()


def load_processor():
    """
        Loads the processor for the given model according to the config and returns . This model does not have a
        processor, so None is returned.
    """

    return None


def preprocess(
        processor,
        question: Union[Tuple[str], List[Tuple[str]]],
        answer: str,
        image: Union[Path, List[Path]]
) -> Dict:
    """
        For the given processor, image, question and answer, preprocess the data into the format used by the model.
        Also, implement additional preprocessing steps here.

        :param processor: The processor to use. Defaults to None in this case.
        :param question: The question and question_id, alone or as a list of strings
        :param answer: The annotated answer to the question as a string
        :param image: The image as a Path to the Image, or a List of Paths.
        :return: The preprocessed data as a Dictionary.
    """

    return dict(questions=[q[1] for q in question],
                question_ids=[q[0] for q in question],
                images=image)


class BEiT3WrapperForEncapsulators():
    """
        Wrapper Class for the BEiT3 Model to simulate a huggingface.transformer class so that this model is compatible
        with the already existing architecture for huggingface transformers.
    """

    def __init__(self):

        # Cast to device
        self.device = torch.device("cpu")

        # Get model config and create model
        self.model = create_model(
            "%s_%s" % (CONFIG["MODEL"], "vqav2"),
            pretrained=False,
            drop_path_rate=CONFIG["DROP_PATH"],
            vocab_size=CONFIG["VOCAB_SIZE"],
            checkpoint_activations=False,
        )

        # Don't fully understand this stuff :)
        utils.load_model_and_may_interpolate(CONFIG["FINETUNE"], self.model, CONFIG["MODEL_KEY"], CONFIG["MODEL_PREFIX"])
        self.model.to(self.device)

        # Set up task handler
        self.task_handler = get_handler(CONFIG)

    def __call__(self, questions, question_ids, images):

        # Create data loader for the passed inputs
        data_loader_test = create_downstream_dataset(
            CONFIG,
            questions=questions,
            question_ids=question_ids,
            img_paths=images
        )

        # Call the evaluate function and return the output
        result, _ = evaluate(data_loader_test, self.model, self.device, self.task_handler)
        return result
