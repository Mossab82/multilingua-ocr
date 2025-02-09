import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, Optional

class DocumentPreprocessor:
    """Document preprocessing pipeline as described in Section 3.1."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 768),
        analyze_degradation: bool = True,
        remove_background: bool = True,
        normalize_contrast: bool = True
    ):
        self.target_size = target_size
        self.analyze_degradation = analyze_degradation
        self.remove_background = remove_background
        self.normalize_contrast = normalize_contrast
        
        # Initialize components
        self.degradation_analyzer = DegradationAnalyzer()
    
    def __call__(self, image: Image.Image) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """Process document image and return processed image with degradation map."""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Analyze degradation if enabled
        degradation_map = None
        if self.analyze_degradation:
            degradation_map = self.degradation_analyzer(img_array)
        
        # Remove background
        if self.remove_background:
            img_array = self._remove_background(img_array)
        
        # Normalize contrast
        if self.normalize_contrast:
            img_array = self._normalize_contrast(img_array)
        
        # Resize to target size
        img_array = cv2.resize(img_array, self.target_size)
        
        return Image.fromarray(img_array), degradation_map
    
    def _remove_background(self, image: np.ndarray) -> np.ndarray:
        """Remove background noise from document image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cv2.bitwise_and(image, image, mask=cleaned)
    
    def _normalize_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast while preserving document features."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

class DegradationAnalyzer:
    """Analyzes document degradation as described in Section 3.1."""
    
    def __init__(
        self,
        threshold: float = 0.3,
        window_size: int = 16
    ):
        self.threshold = threshold
        self.window_size = window_size
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Generate degradation map for document image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate local variance as degradation indicator
        local_var = self._compute_local_variance(gray)
        
        # Normalize and threshold
        degradation_map = (local_var - local_var.min()) / (local_var.max() - local_var.min())
        degradation_map = (degradation_map > self.threshold).astype(np.float32)
        
        return degradation_map
    
    def _compute_local_variance(self, image: np.ndarray) -> np.ndarray:
        """Compute local variance for degradation detection."""
        kernel = np.ones((self.window_size, self.window_size)) / (self.window_size ** 2)
        mean = cv2.filter2D(image.astype(float), -1, kernel)
        mean_sq = cv2.filter2D(image.astype(float)**2, -1, kernel)
        return mean_sq - mean**2

class ScriptDetector:
    """Detects and classifies scripts for attention masking."""
    
    def __init__(self, confidence_threshold: float = 0.85):
        self.confidence_threshold = confidence_threshold
        
    def create_mask(
        self,
        text: str,
        language_pair: Tuple[str, str]
    ) -> torch.Tensor:
        """Create attention mask based on script detection."""
        # Convert text to tokens
        tokens = text.split()
        n = len(tokens)
        
        # Initialize mask
        mask = torch.ones(n, n)
        
        # Detect script for each token
        scripts = [self._detect_script(token, language_pair) for token in tokens]
        
        # Create compatibility mask
        for i, script_i in enumerate(scripts):
            for j, script_j in enumerate(scripts):
                if script_i != script_j:
                    mask[i, j] = 0
        
        return mask
    
    def _detect_script(self, token: str, language_pair: Tuple[str, str]) -> str:
        """Detect script of a token based on character properties."""
        # Implement script detection logic based on Unicode ranges
        # and language-specific patterns
        pass
