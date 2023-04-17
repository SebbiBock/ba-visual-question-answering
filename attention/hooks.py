"""
    File that contains classes that are used to register methods to forward hooks
    in order to e.g. extract information such as attention weights.
"""

import torch


class EncapsulateTransformerAttention(object):
    """
        EncapsulateTransformerAttention encapsulates a PyTorch Transformer and registers a method to a forward hook
        in order to save attention weights at every of the provided layers.
    """

    def __init__(self, model, attention_layer_name):
        """
            Constructor to encapsulate a given model according to the given attention layer name.

            :param model: The Transformer to encapsulate.
            :param attention_layer_name: String that a layer name needs to contain so that a forward hook is registered.
        """

        self.model = model

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        # Saves the attentions
        self.attentions = []

    def __call__(self, **kwargs) -> tuple:
        """
            Calls the model with the registered hooks for inference and returns the
            output along with the attention weights.

            :param kwargs: Necessary arguments for the model call, are unpacked.
        """

        self.attentions = []

        with torch.no_grad():
            output = self.model(**kwargs)

        return output, self.attentions

    def get_attention(self, module, input, output):
        """
            Attention getter to register to a hook.
        """

        self.attentions.append(output.cpu())