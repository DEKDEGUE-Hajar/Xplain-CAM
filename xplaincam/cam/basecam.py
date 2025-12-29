from __future__ import annotations

import torch
import torch.nn as nn

class BaseCAM:
    """
    Base class for Class Activation Map (CAM) methods.

    This class handles:
        - Registering forward/backward hooks to capture activations and gradients
        - Automatic resolution of target convolutional layers
        - Optional fully connected layer resolution for methods like GradCAM++
        - Context manager support (with ... as)

    Subclasses must implement the `forward` method to compute the CAM.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | None = None,
        input_shape: tuple[int, int, int] = (3, 224, 224)  # CHW format
    ):
        """
        BaseCAM constructor.

        Args:
            model: PyTorch model
            target_layer: layer name or module to hook for CAM. If None, automatically
                        selects the last conv layer.
            input_shape: shape of input tensor (C, H, W). Used by some CAMs to initialize
                        masks or upsample sizes.
        """
        self.model = model.eval()
        self.input_shape = input_shape
        self.activations = {}
        self.gradients = {}
        self._hooks = []

        # --- resolve target layer ---
        if target_layer is None:
            conv_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
            if not conv_layers:
                raise ValueError("No convolutional layers found in model. Please specify `target_layer`.")
            target_layer_name = conv_layers[-1]
            # print(f"[BaseCAM] No target_layer provided, using last conv layer: '{target_layer_name}'")
        elif isinstance(target_layer, str):
            target_layer_name = target_layer
        elif isinstance(target_layer, nn.Module):
            target_layer_name = self._resolve_layer_name(target_layer)
        else:
            raise TypeError("target_layer must be None, str, or nn.Module")

        layer_module = self._get_module_by_name(target_layer_name)
        self._hooks.append(layer_module.register_forward_hook(self._forward_hook))
        self._hooks.append(layer_module.register_backward_hook(self._backward_hook))
        self.target_layer_name = target_layer_name

        # -------------------------------
        # Optional fully connected layer (for CAM variants needing FC weights)
        # -------------------------------
        # if fc_layer is None:
        #     # Try to pick the last Linear layer automatically
        #     fc_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
        #     if fc_layers:
        #         self.fc_layer_name = fc_layers[-1]
        #         print(f"[BaseCAM] No fc_layer provided, using last linear layer: '{self.fc_layer_name}'")
        #     else:
        #         self.fc_layer_name = None
        # elif isinstance(fc_layer, str):
        #     self.fc_layer_name = fc_layer
        # elif isinstance(fc_layer, nn.Module):
        #     self.fc_layer_name = self._resolve_layer_name(fc_layer)
        # else:
        #     raise TypeError("fc_layer must be None, str, or nn.Module")

        # # Store fc weights if applicable
        # if self.fc_layer_name is not None:
        #     fc_module = self._get_module_by_name(self.fc_layer_name)
        #     self._fc_weights = fc_module.weight.data.clone()
        # else:
        #     self._fc_weights = None

    # Hook functions
    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations['value'] = output

    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients['value'] = grad_output[0]

    # Layer helper functions
    def _resolve_layer_name(self, module: nn.Module):
        """Return the name of a module in the model."""
        for name, m in self.model.named_modules():
            if m is module:
                return name
        raise ValueError("Module not found in model.")

    def _get_module_by_name(self, name: str):
        """Retrieve a module from the model given its dotted name."""
        module = self.model
        for attr in name.split('.'):
            if not hasattr(module, attr):
                raise ValueError(f"Module '{attr}' not found in '{module}'.")
            module = getattr(module, attr)
        return module

    # Forward function (must be overridden)
    def forward(self, x: torch.Tensor, class_idx: int | None = None, retain_graph: bool = False):
        """
        Compute the class activation map.

        Args:
            x: input tensor
            class_idx: target class index. If None, uses predicted class
            retain_graph: whether to retain the computation graph for multiple backward passes
        Returns:
            CAM map (tensor)
        """
        raise NotImplementedError

    def __call__(self, x: torch.Tensor, class_idx: int | None = None, retain_graph: bool = False):
        """Allows instance to be called as a function."""
        return self.forward(x, class_idx, retain_graph)

    # Context manager support
    def __enter__(self):
        """Support for 'with ... as'."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove all hooks when exiting context."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
