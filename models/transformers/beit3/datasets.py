# --------------------------------------------------------
# Image as a Foreign Language: BEiT Pretraining for Vision and Vision-Language Tasks (https://arxiv.org/abs/2208.10442)
# Github source: https://github.com/microsoft/unilm/tree/master/beit3
# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import json
import random
import torch
import glob
from collections import defaultdict, Counter
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform
from pathlib import Path

import models.transformers.beit3.utils as utils
from models.transformers.beit3.glossary import normalize_word
import data.loader as loader


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, questions, question_ids, img_paths, transform, tokenizer, num_max_bpe_tokens):
        self.tokenizer = tokenizer
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.items = self._create_own_items(questions, question_ids, img_paths)
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.loader = default_loader
        self.transform = transform

    def _get_image(self, image_path: str):

        # Load in the image
        image = self.loader(Path(image_path))

        # Unsqueeze the image for dummy dimension: We have no batches, so B = 1 needs to be assured.
        return torch.unsqueeze(self.transform(image), 0)

    def _get_text_segment(self, text_segment, max_len=None):
        if isinstance(text_segment, str):
            tokens = self.tokenizer.tokenize(text_segment)
        else:
            tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)

        # Return as tensors and set proper dtype on creation. Also unsqueeze to simulate batches
        return torch.unsqueeze(torch.tensor(tokens + [self.pad_token_id] * (max_len - num_tokens), dtype=torch.long), 0),\
               torch.unsqueeze(torch.tensor(padding_mask, dtype=torch.long), 0), num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __len__(self) -> int:
        return len(self.items)

    def _create_own_items(self, questions, question_ids, img_paths):

        # Construct item list
        items = []

        for question_text, qid, img_path in zip(questions, question_ids, img_paths):

            # Get question tokens
            tokens = self.tokenizer.tokenize(question_text)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            items.append({
                "image_path": img_path,
                "text_segment": token_ids,
                "qid": qid,
                "labels": [],  # Can be empty because no training
                "scores": []  # Can be empty because no training
            })

        return items


class VQAv2Dataset(BaseDataset):
    def __init__(self, questions, question_ids, img_paths, **kwargs):
        super().__init__(questions, question_ids, img_paths, **kwargs)
        self.ans2label = {}
        self.label2ans = {}
        self.get_answer_encodings()

    def __getitem__(self, index: int):
        data = super().__getitem__(index)
        # Append qid to data for clear indexing later
        data["qid"] = torch.tensor([int(self.items[index]["qid"])], dtype=torch.long)

        return data

    def get_answer_encodings(self):

        # Load in annotated data from VQAv2
        with open(Path(loader.PATH_DICT["ANNOTATION_PATH"]), "rb") as f:
            annotations_val = json.load(f)["annotations"]
        with open(Path(loader.PATH_DICT["ANNOTATION_TRAIN_PATH"]), "rb") as f:
            annotations_train = json.load(f)["annotations"]

        all_major_answers = []

        for annots in [annotations_train, annotations_val]:
            for q in annots:
                all_major_answers.append(q["multiple_choice_answer"])

        all_major_answers = [normalize_word(word) for word in all_major_answers]
        counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}

        self.ans2label = {k: i for i, k in enumerate(counter.keys())}
        self.label2ans = list(counter.keys())


def create_dataloader(dataset, batch_size):

    sampler = torch.utils.data.SequentialSampler(dataset)
    
    return torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        #collate_fn=utils.merge_batch_tensors_by_dict_key,
    )


def build_transform(args):

    return transforms.Compose([
        transforms.Resize((args["INPUT_SIZE"], args["INPUT_SIZE"]), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])


def get_sentencepiece_model_for_beit3(args):
    from transformers import XLMRobertaTokenizer
    return XLMRobertaTokenizer(args["SENTENCEPIECE_MODEL"])


def create_dataset_by_split(args, questions, question_ids, img_paths):

    transform = build_transform(args)
    tokenizer = get_sentencepiece_model_for_beit3(args)

    dataset = VQAv2Dataset(
        questions=questions, question_ids=question_ids, img_paths=img_paths,
        transform=transform, tokenizer=tokenizer, num_max_bpe_tokens=args["NUM_MAX_BPE_TOKENS"]
    )

    # Set batch size
    batch_size = args["EVAL_BATCH_SIZE"]

    return create_dataloader(dataset, batch_size=batch_size)


def create_downstream_dataset(args, questions, question_ids, img_paths):
    return create_dataset_by_split(args, questions, question_ids, img_paths)
