import pytest
import numpy as np
from PIL import Image
from multilingua_ocr.data import (
    DocumentPreprocessor,
    DegradationAnalyzer,
    ScriptDetector
)
from multilingua_ocr.core import PreprocessConfig

@pytest.fixture
def sample_image():
    # Create synthetic document image
    image = np.zeros((300, 200, 3), dtype=np.uint8)
    image[50:250, 50:150] = 255  # Add text-like region
    return Image.fromarray(image)

@pytest.fixture
def preprocess_config():
    return PreprocessConfig(
        target_height=224,
        target_width=224,
        normalize_contrast=True,
        remove_background=True
    )

def test_document_preprocessor(sample_image, preprocess_config):
    processor = DocumentPreprocessor(**preprocess_config.__dict__)
    processed_image, degradation_map = processor(sample_image)
    
    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (
        preprocess_config.target_width,
        preprocess_config.target_height
    )
    
    if degradation_map is not None:
        assert isinstance(degradation_map, np.ndarray)
        assert degradation_map.shape[:2] == (
            preprocess_config.target_height,
            preprocess_config.target_width
        )

def test_degradation_analyzer(sample_image):
    analyzer = DegradationAnalyzer()
    degradation_map = analyzer(np.array(sample_image))
    
    assert isinstance(degradation_map, np.ndarray)
    assert degradation_map.shape[:2] == sample_image.size[::-1]
    assert np.all(degradation_map >= 0) and np.all(degradation_map <= 1)

def test_script_detector():
    detector = ScriptDetector()
    text = "El tlatoani mandó que todos los tepaneca prepararan sus yaotlatquitl"
    language_pair = ("Spanish", "Nahuatl")
    
    mask = detector.create_mask(text, language_pair)
    
    assert isinstance(mask, torch.Tensor)
    words = text.split()
    assert mask.shape == (len(words), len(words))
