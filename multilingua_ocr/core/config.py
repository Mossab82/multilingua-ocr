from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import yaml

@dataclass
class ModelConfig:
    """Configuration for the MultiLingua-OCR model architecture."""
    
    # Model Architecture Parameters
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout_rate: float = 0.1
    
    # Script-Aware Attention Parameters
    use_script_attention: bool = True
    script_embedding_dim: int = 64
    max_sequence_length: int = 512
    
    # Cultural-Semantic Parameters
    cultural_embedding_dim: int = 256
    ontology_size: int = 1000
    language_pairs: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.language_pairs is None:
            self.language_pairs = [
                ("Spanish", "Nahuatl"),
                ("Spanish", "Mixtec"),
                ("Spanish", "Zapotec")
            ]
    
    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.__dict__, f)

@dataclass
class TrainingConfig:
    """Configuration for model training parameters."""
    
    # Basic Training Parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Optimizer Parameters
    optimizer: str = "AdamW"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    
    # Loss Weights
    ocr_loss_weight: float = 1.0
    cultural_loss_weight: float = 0.5
    script_loss_weight: float = 0.3
    
    # Performance Thresholds
    min_cer: float = 0.15  # Maximum acceptable Character Error Rate
    min_sps: float = 0.8   # Minimum Semantic Preservation Score
    min_cca: float = 0.8   # Minimum Cultural Concept Accuracy

@dataclass
class PreprocessConfig:
    """Configuration for document preprocessing."""
    
    # Image Processing Parameters
    target_height: int = 1024
    target_width: int = 768
    normalize_contrast: bool = True
    remove_background: bool = True
    
    # Degradation Analysis
    analyze_degradation: bool = True
    degradation_threshold: float = 0.3
    
    # Data Augmentation
    use_augmentation: bool = True
    augmentation_probability: float = 0.5
    max_rotation_angle: float = 5.0
    
    # Script Detection
    min_confidence_threshold: float = 0.85
    overlap_threshold: float = 0.5
