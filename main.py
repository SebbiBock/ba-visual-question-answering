import matplotlib.pyplot as plt
import numpy as np

import models.transformers.vilt as vilt
import models.transformers.blip as blip
import data.loader as loader
import util.data_util as dutil

from attention.attention_rollout import attention_rollout
from attention.grad_cam import compute_grad_cam_for_layers
from attention.hooks import EncapsulateTransformerAttention, EncapsulateTransformerActivationAndGradients
from plotting.plot_comparison import plot_overview_for_question
from util.image import fuze_image_and_array, resize_array_to_img


if __name__ == '__main__':

    # Choose VQAv2 question IDs as strings
    # question_ids = ["109945002"]

    # Randomly chosen 20 ones
    question_ids = ['566550013', '491061001', '214224017', '565273001', '368576000',
                    '312552002', '67342003', '174070000', '77891000', '536403008',
                    '489733005', '497568003', '445233002', '342318002', '213592002',
                    '217517001', '523527002', '384531001', '568972001', '109945002']

    # Choose model
    model_package = blip

    # Create output path, if it doesn't exist
    dutil.construct_output_folder()

    # Load model and preprocessor
    processor = model_package.load_processor()
    model = model_package.load_model()

    # Load in the necessary data
    images = loader.load_images(question_ids)
    questions = loader.load_questions(question_ids)
    annotated_answers = loader.load_annotated_answers(question_ids, single_answer=True)
    human_answers_df = loader.load_human_answers()

    # Encapsulate model and choose the proper attention layer for the forward hook registration
    encapsulated_model = EncapsulateTransformerAttention(
        model,
        model_package.CONFIG["ATTENTION_LAYER_HOOK_NAME"],
        unsqueeze_attentions=model_package.CONFIG["UNSQUEEZE_ATTENTIONS"],
        model_wrapper_used=model_package.CONFIG["MODEL_WRAPPER_USED"]
    )

    # Encapsulated model for the gradients and activations of the target layers
    encapsulated_gradient_model = EncapsulateTransformerActivationAndGradients(model, [
        model.vision_model.encoder.layers[-2].layer_norm2
    ])

    # List of heatmaps to be generated
    grad_cam_heatmaps = []
    att_heatmaps = []
    human_heatmaps = []
    model_answers = []
    human_answers = []

    # For every image-question pair, get gradients and activations
    for image, question, q_id in zip(images, questions, question_ids):

        # Preprocess the data for the model
        model_input = model_package.preprocess(processor, question, annotated_answers[q_id], image)

        # Inference: Get activations and gradients w.r.t. the predicted output class
        encapsulated_gradient_model(**model_input)

        # For now, we only take the first one of the registered layers (only really one necessary)
        grad_cam_heatmap = compute_grad_cam_for_layers(
            encapsulated_gradient_model.gradients, encapsulated_gradient_model.activations
        )[0]

        # Resize to image and fuze
        resized_grad_heatmap = resize_array_to_img(image, grad_cam_heatmap)
        grad_cam_heatmaps.append(fuze_image_and_array(image, resized_grad_heatmap))

        # Get mean human heatmap, resize and fuze
        human_heatmap = loader.load_human_heatmaps(q_id, reduction=np.mean)
        resized_human_heatmap = resize_array_to_img(image, human_heatmap)
        human_heatmaps.append(fuze_image_and_array(image, resized_human_heatmap))

        # Get human answers
        human_answers.append(dutil.get_answers_for_question(human_answers_df, q_id))

    # Release hooks so that second encapsulator works
    encapsulated_gradient_model.release()

    # For every image-question pair, get attentions
    for image, question, q_id in zip(images, questions, question_ids):

        # Preprocess the data for the model
        model_input = model_package.preprocess(processor, question, annotated_answers[q_id], image)

        # Inference / Forward: Get output and attentions from the encapsulated model
        output, attentions = encapsulated_model(**model_input)

        # Get model answers: Some of these parameters might not be necessary, but for consistency across models needed
        model_answers.append(model_package.process_output(
            model,
            processor,
            **output,
            **model_input
        ))

        # Get heatmap(s) for model attention
        rollout_attention_map = attention_rollout(attentions, head_fusion="max")

        # Resize heatmap and fuze with image
        resized_heatmap = resize_array_to_img(image, rollout_attention_map)
        att_heatmaps.append(fuze_image_and_array(image, resized_heatmap))

    # Release hooks
    encapsulated_model.release()

    # Create plot for every question-image pair to compare heatmaps
    for idx, q_id in enumerate(question_ids):

        plot_overview_for_question(
            human_heatmap=human_heatmaps[idx],
            grad_cam_heatmap=grad_cam_heatmaps[idx],
            att_rollout_heatmap=att_heatmaps[idx],
            human_answers=human_answers[idx],
            model_answer=model_answers[idx],
            question=questions[idx],
            question_id=q_id
        )

