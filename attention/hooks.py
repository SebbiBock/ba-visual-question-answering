"""
    File that contains classes that are used to register methods to forward hooks
    in order to e.g. extract information such as attention weights.
"""

import torch

from typing import List


class EncapsulateTransformerAttention(object):
    """
        EncapsulateTransformerAttention encapsulates a PyTorch Transformer and registers a method (handle) to a
        forward hook in order to save attention activations at every of the provided layers.
    """

    def __init__(self, model, attention_layer_name: str):
        """
            Constructor to encapsulate a given model according to the given attention layer name.

            :param model: The Transformer to encapsulate.
            :param attention_layer_name: String that a layer name needs to contain so that a forward hook is registered.
        """

        self.model = model

        # Save all handles to be able to release them
        self.handles = []

        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                self.handles.append(module.register_forward_hook(self.get_attention))

        # Saves the attentions
        self.attentions = []

    def __call__(self, **kwargs) -> tuple:
        """
            Calls the model with the registered hooks for inference and returns the
            output along with the attention activation.

            :param kwargs: Necessary arguments for the model call, are unpacked.
            :return: Tuple of output and attention list
        """

        # Reset at each call if this class is used multiple times
        self.attentions = []

        with torch.no_grad():
            output = self.model(**kwargs)

        return output, self.attentions

    def get_attention(self, module, input, output):
        """
            Attention getter handle to register to a hook.
        """

        self.attentions.append(output.cpu())

    def release(self):
        """
            Release all registered handles.
        """

        for handle in self.handles:
            handle.remove()


class EncapsulateTransformerActivationAndGradients(object):
    """
        EncapsulateTransformerAttentionAndGradients encapsulates a PyTorch Transformer and registers a method (handle)
        to a forward hook in order to save activations and gradients at every of the provided layers.
    """

    def __init__(self, model, target_layers: List[torch.nn.Sequential], transform: callable = None):
        """
            Constructor to encapsulate a given model according to the given layer name.

            :param model: The Transformer to encapsulate.
            :param target_layers: List of layers that the handles should be registered to.
            :param transform: Callable function that takes the activations / gradients as an input to transform them, defaults to None
        """

        self.model = model
        self.gradients = []
        self.activations = []
        self.transform = transform

        # Save all handles to be able to release them
        self.handles = []

        # For every target layer, register the handles and save them to the handle list
        for layer in target_layers:
            self.handles.append(layer.register_forward_hook(self.save_activation))
            self.handles.append(layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        """
            Activation getter handle that saves the activation and transforms it, if necessary.
        """

        if self.transform is not None:
            activation = self.transform(output)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        """
            Gradient getter handle that saves the gradient and transforms it, if necessary
        """

        def _store_grad(grad):
            """
                Helper method (handle) that stores the gradients, in reverse order
            """
            if self.transform is not None:
                grad = self.transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        # Register the helper hook to the output gradient
        output.register_hook(_store_grad)

    def __call__(self, **kwargs) -> tuple:
        """
            Calls the model with the registered hooks for inference and returns the
            output along with the attention activation.

            :param kwargs: Necessary arguments for the model call, are unpacked.
            :return: Tuple of model output, gradient and activations
        """

        # Reset at each call if this class is used multiple times
        self.gradients = []
        self.activations = []

        return self.model(**kwargs), self.gradients, self.activations

    def release(self):
        """
            Release all registered handles.
        """

        for handle in self.handles:
            handle.remove()
