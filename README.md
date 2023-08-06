# Bachelor Thesis: Human and Transformer Attention in Visual Question Answering

This is the repository for my Bachelor's thesis. In order to run all code, you need to download the validation split of the VQAv2 dataset (https://visualqa.org/) and adjust the corresponding paths in data.loader. Furthermore, you'll need to download pretrained model weights for BEiT3 from this repository (https://github.com/microsoft/unilm/tree/master/beit3) and place it into models.transformers.beit3.pretrained.

To run inference with any model, run the provided code in compue_model_predictions.py. The subsequent evaluation code can be found in compute_performance.py. Finally, the data analysis steps can be found in the respective evaluation notebook.

The entire experiment along with precise step-by-step instructions can be found under /experiment. However, due to data privacy regulations, I did not upload any participant data.
