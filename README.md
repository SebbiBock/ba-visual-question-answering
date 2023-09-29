# Bachelor Thesis: Human and Transformer Attention in Visual Question Answering


## Abstract

Visual Question Answering (VQA) is a cognitively demanding multi-modal task, where visual and linguistic information must be integrated to provide answers to inquiries regarding image content. The recent emergence of transformers yields models capable of achieving human-like proficiency in such vision and language tasks. One mechanism of transformers adapted from humans that is crucial to their advancement is attention. We conducted an eye-tracking experiment with 20 participants and three SOTA-transformers to examine similarities of attention in VQA while mitigating various structural biases and incorporating common reasoning categories. Although humans and models exhibit high proficiency in VQA, significant differences in transformer performances between reasoning types were disclosed, alluding to inflated performance scores of previous literature due to the skewed distribution typically favoring trivial tasks. The recorded human fixations revealed a center bias along with a vertical preference, and variance of fixation locations in-between reasoning types could be found and exploited as moderate indicators for human performance. Qualitative analyses of attention heatmaps unveiled that humans frequently attend to high-level visual information such as faces, whereas transformers favor low-level features like brightness contrasts. Despite their competitive results, we show that transformers generally do not attend to the same image regions as humans do, and a quantitative evaluation resulted in low similarity ratings. Contrary to previous expectations, we found similar attendance to deteriorate transformer performance in VQA, indicating that the attention mechanism of humans operates under vastly different properties. Subsequent inquiries regarding the eligibility of different measurements as predictors of human and transformer proficiency disclosed dwell times, blink rates and reasoning categories to hold significant influences on performance scores.

## Setup

To allow for easy usage of this repository, all data paths are gathered in ```data/data_loader.py```. Here, you'll need to adjust ```DATA_PATH``` to the folder where your data is stored. In order to run the code, you need to follow these subsequent steps:

1. Setup a virtual environment based on ```requirements.txt```.
2. Download the validation split of the [VQAv2 dataset](https://visualqa.org/) and adjust the corresponding paths to images, questions and annotations.
3. Download pretrained model weights for BEiT3 from [this repository](https://github.com/microsoft/unilm/tree/master/beit3) and place it into ```models.transformers.beit3.pretrained```.

## Inference

To run inference with any model, run the provided code in ```compue_model_predictions.py``` and set the ```model_package``` to the preferred model. The subsequent evaluation code can be found in compute_performance.py. The data analysis steps can be found in the respective evaluation notebook.

## Experiment

The entire experimental code and setup, along with precise step-by-step instructions, can be found under ```/experiment```. However, due to data privacy regulations, participant data is not provided alongside with this repository.

