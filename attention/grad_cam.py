"""
    This file contains necessary code to compute the Gradient-weighted Class Activation Mapping (Grad-CAM). For further
    information, see https://arxiv.org/pdf/1610.02391.pdf.
"""

import torch

import numpy as np

from typing import List, Tuple, Union


def transformer_reshape_transform(tensor, height, width) -> torch.Tensor:
    """
        Reshape the activation and gradient tensors to match that of CNN networks. Idea taken from
        https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html: "In ViT the output
        of the layers are typically BATCH x 197 x 192. In the dimension with 197, the first element
        represents the class token, and the rest represent the 14x14 patches in the image. We can
        treat the last 196 elements as a 14x14 spatial image, with 192 channels".

        Same principle as in attention rollout!

        :param tensor: Tensor of activations or gradients from a vision transformer
        :param height: Amount of image patches in height
        :param width: Amount of image patches in width
        :return: Reshaped torch.Tensor
    """

    # Reshape the result to the given height and width of image patches and dis-regard the class token
    # Shape should be: (batch_size, amount_patches_h, amount_patches_w, channels)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring channels to first dimensions to get tensor shape equal to that in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def compute_grad_cam_for_layers(
    gradients: List[torch.Tensor],
    activations: List[torch.Tensor],
    text_embed_length: int,
    amount_image_patches: Tuple[int, int],
    image_patch_embedding_retrieval_fct: Union[callable, None],
    reshape_transform: callable = transformer_reshape_transform
) -> List[np.array]:
    """
        Compute the Gradient-weighted Class Activation Mapping (Grad-CAM) from the given gradients and activations.
        If the output comes from a Vision Transformer (ViT), a reshape_transform function needs to be provided in
        order to reshape the Tensors to the proper shape given in CNNs. The Grad-CAM map is computed for every layer
        that was registered, so the inputs need to be lists, each element representing one Tensor taken from a
        hooked module layer.

        The encapsulated model where the gradients and activations stem from can be gotten at every layer of the
        module for every possible output class. Mostly, we take the predicted class (model answer), and compute
        the Grad-CAM map w.r.t that output. Therefore, a backward call on the loss from the label to the currently
        viewed class needs to be propagated to get the proper gradients. This is done in hooks.py.

        :param gradients: List of torch.Tensors, representing the gradients for loss.backward() at every hooked layer
        :param activations: List of torch.Tensors, representing the activations at every hooked layer
        :param text_embed_length: Length of the textual embedding of the question, so how many tokens are used.
        :param amount_image_patches: The amount of image patches for the given model and image in height and width dimension
        :param image_patch_embedding_retrieval_fct: Callable function to reduce the attention matrix to only attention on the image embeddings, if attention is deployed on the textual input. Since the order of visual and textual input is dependent on the model, this is outsourced.
        :param reshape_transform: Callable function that transform the gradients and activations to proper shape.
        :return: Grad-CAM map for every hooked layer w.r.t the given loss of the chosen class
    """

    # Result list
    grad_cam_heatmaps = []

    for grad, act in zip(gradients, activations):

        # Detach grad and cast to cpu to avoid manipulation of actual Tensors
        reshaped_grad = grad.detach().cpu()
        reshaped_act = act.detach().cpu()

        # Remove entries of textual embeddings to retrieve pure image activations and gradients
        if image_patch_embedding_retrieval_fct is not None:
            reshaped_grad = image_patch_embedding_retrieval_fct(reshaped_grad, text_embed_length)
            reshaped_act = image_patch_embedding_retrieval_fct(reshaped_act, text_embed_length)

        # Reshape gradient and activations to proper form (similar to CNN), if the input stems from a ViT
        if reshape_transform is not None:
            reshaped_grad = reshape_transform(reshaped_grad, amount_image_patches[0], amount_image_patches[1])
            reshaped_act = reshape_transform(reshaped_act, amount_image_patches[0], amount_image_patches[1])

        # Pool the gradients across the channels
        pooled_gradients = torch.mean(reshaped_grad, dim=[0, 2, 3])

        # Weight the channels by corresponding gradients
        for i in range(reshaped_grad.shape[1]):
            reshaped_act[:, i, :, :] *= pooled_gradients[i]

        # Average the channels of the activations
        heatmap = torch.mean(reshaped_act, dim=1).squeeze()

        # Perform ReLU on the heatmap
        heatmap = np.maximum(heatmap, 0)

        # Normalize the heatmap
        heatmap /= torch.max(heatmap)

        # Transform heatmap tensor to numpy array and append to list of results
        grad_cam_heatmaps.append(heatmap.squeeze().detach().numpy())

    return grad_cam_heatmaps
