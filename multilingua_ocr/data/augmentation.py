import random
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

class DocumentAugmenter:
    """Implements data augmentation techniques as described in Section 3.3."""
    
    def __init__(
        self,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_probability: float = 0.3,
        degradation_simulation: bool = True
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.contrast_range = contrast_range
        self.noise_probability = noise_probability
        self.degradation_simulation = degradation_simulation
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentations to document image."""
        # Convert to tensor for transformations
        img_tensor = TF.to_tensor(image)
        
        # Geometric transformations
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            img_tensor = TF.rotate(img_tensor, angle)
        
        # Scale augmentation
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            new_size = [int(s * scale) for s in img_tensor.shape[-2:]]
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=new_size,
                mode='bilinear'
            ).squeeze(0)
        
        # Contrast adjustment
        if random.random() < 0.5:
            contrast = random.uniform(*self.contrast_range)
            img_tensor = TF.adjust_contrast(img_tensor, contrast)
        
        # Add noise to simulate degradation
        if self.degradation_simulation and random.random() < self.noise_probability:
            noise = torch.randn_like(img_tensor) * 0.1
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        
        return TF.to_pil_image(img_tensor)
