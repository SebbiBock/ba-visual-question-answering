"""
    Class for attention rollout methods, inspired by
    https://jacobgil.github.io/deeplearning/vision-transformer-explainability.
"""

import torch
import numpy as np

from typing import Tuple, Union


def attention_rollout(
        atts,
        image_patch_embedding_retrieval_fct: Union[callable, None],
        text_embed_length: int,
        amount_image_patches: Tuple[int, int],
        discard_ratio=0.85,
        head_fusion="max"
) -> np.array:
    """
        AttentionRollout method: Compute heatmaps from transformer attention head weights.

        :param atts: List of attention weights.
        :param image_patch_embedding_retrieval_fct: Callable function to reduce the attention matrix to only attention on the image embeddings, if attention is deployed on the textual input. Since the order of visual and textual input is dependent on the model, this is outsourced.
        :param text_embed_length: Length of the textual embedding of the question, so how many tokens are used.
        :param amount_image_patches: The amount of image patches for the given model and image in height and width dimension
        :param discard_ratio: Percentage of the weakest pixels that are to be discarded.
        :param head_fusion: String identifier for how to fuse the different attention heads together: ["mean", "min", "max"]
        :return: Attention heatmap as a numpy array.
    """

    # Detach and clone the Tensors to avoid performing operations on original Tensors
    attentions = [x.detach().clone() for x in atts]

    # If the model uses the attention mechanism on the text input as well, we need to remove the text token
    # embeddings from the given attentions to only look at the visual patch embeddings.
    if image_patch_embedding_retrieval_fct is not None:
        for idx, a in enumerate(attentions):

            # Throw out attention on text tokens according to the embedding of the model.
            attentions[idx] = image_patch_embedding_retrieval_fct(a, text_embed_length)

    # Build-up result matrix that needs to be the shape of (p^2 + text_embed_tokens + 1)^2
    result = torch.eye(attentions[0].size(-1))

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Unsupported head fusion!"

            # Drop the lowest attentions according to the discard ratio, but make sure to keep the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            # Add identity matrix to account for residual connections
            I = torch.eye(attention_heads_fused.size(-1))

            # Re-normalize the weights
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1)

            result = torch.matmul(a, result)

    # Look at the total attention only between the class token to all the image patches, this gives us
    # one vector of size (amount_patches ^ 2).
    mask = result[0, 0, 1:]

    # Get amount of image patches
    amt_patches_height, amt_patches_width = amount_image_patches

    # Reshape the vector to a amt_patches_height x amt_patches_width image (heatmap)
    mask = mask.reshape(amt_patches_height, amt_patches_width).numpy()
    return mask / np.max(mask)
