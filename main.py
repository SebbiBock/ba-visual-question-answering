import matplotlib.pyplot as plt

import models.transformers.vilt as vilt
import models.transformers.blip as blip
import data.loader as loader

from attention.attention_rollout import attention_rollout
from attention.hooks import EncapsulateTransformerAttention
from util.image import fuze_image_and_array, resize_array_to_img


if __name__ == '__main__':

    # Choose VQAv2 question IDs as strings
    question_ids = ["109945002"]

    # Choose model
    model_package = blip

    # Load model and preprocessor
    processor = model_package.load_processor()
    model = model_package.load_model()

    # Load in the necessary data
    images = loader.load_images(question_ids)
    questions = loader.load_questions(question_ids)
    annotated_answers = loader.load_annotated_answers(question_ids, single_answer=True)

    # Encapsulate model and choose the proper attention layer for the forward hook registration
    encapsulated_model = EncapsulateTransformerAttention(model, model_package.CONFIG["ATTENTION_LAYER_HOOK_NAME"])

    # For every image-question pair
    for image, question, q_id in zip(images, questions, question_ids):

        # Preprocess the data for the model
        model_input = model_package.preprocess(processor, question, annotated_answers[q_id], image)

        # Inference / Forward: Get output and attentions from the encapsulated model
        output, attentions = encapsulated_model(**model_input)

        # Get heatmap(s) for model attention
        rollout_attention_map = attention_rollout(attentions, head_fusion="max")

        # Resize heatmap and fuze with image
        resized_heatmap = resize_array_to_img(image, rollout_attention_map)
        fuzed_heatmap = fuze_image_and_array(image, resized_heatmap)

        plt.imshow(fuzed_heatmap)

    plt.show()




