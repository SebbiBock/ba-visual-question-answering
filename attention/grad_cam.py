"""
    This file contains necessary code to compute the Gradient-weighted Class Activation Mapping (Grad-CAM). For further
    information, see https://arxiv.org/pdf/1610.02391.pdf.
"""

import torch

import numpy as np

from typing import List


def transformer_reshape_transform(tensor, height=24, width=24) -> torch.Tensor:
    """
        Reshape the activation and gradient tensors to match that of CNN networks. Idea taken from
        https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html: "In ViT the output
        of the layers are typically BATCH x 197 x 192. In the dimension with 197, the first element
        represents the class token, and the rest represent the 14x14 patches in the image. We can
        treat the last 196 elements as a 14x14 spatial image, with 192 channels".

        :param tensor: Tensor of activations or gradients from a vision transformer
        :param height: Height of the patches in the image
        :param width: Width of the patches in the image
        :return: Reshaped torch.Tensor
    """

    # Reshape the result to the given height and width of image patches and dis-regard the class token
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring channels to first dimensions to get tensor shape equal to that in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def compute_grad_cam_for_layers(
    gradients: List[torch.Tensor],
    activations: List[torch.Tensor],
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
        :param reshape_transform: Callable function that transform the gradients and activations to proper shape.
        :return: Grad-CAM map for every hooked layer w.r.t the given loss of the chosen class
    """

    # Result list
    grad_cam_heatmaps = []

    for grad, act in zip(gradients, activations):

        # Detach grad and cast to cpu to avoid manipulation of actual Tensors
        reshaped_grad = grad.detach().cpu()
        reshaped_act = act.detach().cpu()

        # Reshape gradient and activations to proper form (similar to CNN), if the input stems from a ViT
        if reshape_transform is not None:
            reshaped_grad = reshape_transform(grad)
            reshaped_act = reshape_transform(act)

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
