"""
    Class for attention rollout methods, inspired by
    https://jacobgil.github.io/deeplearning/vision-transformer-explainability.
"""

import torch
import numpy as np


def attention_rollout(attentions, discard_ratio=0.85, head_fusion="max") -> np.array:
    """
        AttentionRollout method: Compute heatmaps from transformer attention head weights.

        :param attentions: List of attention weights.
        :param discard_ratio: Percentage of the weakest pixels that are to be discarded.
        :param head_fusion: String identifier for how to fuse the different attention heads together: ["mean", "min", "max"]
        :return: Attention heatmap as a numpy array.
    """

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

    # Look at the total attention between the class token and the image patches
    mask = result[0, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)

    return mask