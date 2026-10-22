import cv2
import numpy as np
import torch
import torch.nn as nn


def find_last_conv_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("No Conv2d layer found in the model")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def __call__(self, tensor, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            x = tensor.clone().requires_grad_(True)
            out = self.model(x)
            if class_idx is None:
                class_idx = int(out.argmax(1).item())
            out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1)).squeeze(0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def close(self):
        self._handle.remove()


def overlay_heatmap(image_rgb, heatmap, alpha=0.4):
    h, w = image_rgb.shape[:2]
    heat = cv2.resize(heatmap, (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heat, alpha, 0)