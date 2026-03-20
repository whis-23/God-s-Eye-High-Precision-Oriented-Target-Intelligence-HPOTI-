import numpy as np
import cv2
import torch

class SARGradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for SAR signature validation.
    Ensures the model focuses on target backscatter rather than land noise.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks for gradients and activations
        def save_gradients(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        def save_activations(module, input, output):
            self.activations = output
            
        target_layer.register_forward_hook(save_activations)
        target_layer.register_backward_hook(save_gradients)

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int):
        """
        Generate heatmap for a specific class activation.
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        # Zero grads
        self.model.zero_grad()
        
        # Target score for class_idx
        score = output[0, class_idx] if len(output.shape) > 1 else output[0]
        score.backward()
        
        # 1. Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 2. Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # 3. ReLU and normalization
        cam = torch.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        # 4. Colorize
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        return heatmap

if __name__ == "__main__":
    import sys
    import os
    # Add models directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
    from gods_eye import GodsEye
    model = GodsEye()
    # Using the last stage of backbone as target layer
    target = model.backbone['stage5']
    grad_cam = SARGradCAM(model, target)
    
    dummy_img = torch.randn(1, 3, 640, 640)
    heatmap = grad_cam.generate_heatmap(dummy_img, class_idx=0)
    print(f"Grad-CAM heatmap generated with shape: {heatmap.shape}")
