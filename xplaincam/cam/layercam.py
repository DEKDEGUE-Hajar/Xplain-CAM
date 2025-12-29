import torch
import torch.nn.functional as F
from typing import Optional, Union, List
from .basecam import BaseCAM


class LayerCAM(BaseCAM):
    """
    LayerCAM: Exploring Hierarchical Class Activation Maps for Localization

    Reference:
        Jiang et al., LayerCAM: Exploring Hierarchical Class Activation Maps for Localization.
        http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf

    """

    def __init__(self, model, target_layer,input_shape=(3, 224, 224), gamma: float = 2.0):
        super().__init__(model, target_layer,input_shape)
        self.gamma = gamma
        self.model.eval()

    def forward(self, x, class_idx: Optional[Union[int, List[int]]] = None, 
                retain_graph: bool = False):

        if x.shape[0] != 1:
            raise ValueError("LayerCAM only supports batch size of 1")

        _, _, h, w = x.size()

        # Forward
        output = self.model(x)

        # Class selection
        if class_idx is None:
            class_idx = [torch.argmax(output).item()]
        elif isinstance(class_idx, int):
            class_idx = [class_idx]

        cams = []

        for idx in class_idx:
            self.model.zero_grad()

            one_hot = torch.zeros_like(output)
            one_hot[0, idx] = 1.0

            output.backward(gradient=one_hot, retain_graph=retain_graph)

            activations = self.activations['value']   # [1, C, H, W]
            gradients = self.gradients['value']       # [1, C, H, W]

            weights = F.relu(gradients)
            cam = torch.sum(weights * activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min)

            cams.append(cam.detach().cpu())

        return cams[0] if len(cams) == 1 else torch.stack(cams)

    def __call__(self, x, class_idx=None, retain_graph=False):
        return self.forward(x, class_idx, retain_graph)

