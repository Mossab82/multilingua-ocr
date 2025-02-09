Core API Reference
Configuration
The core module provides configuration classes for model architecture, training, and preprocessing settings.
ModelConfig
Configuration for model architecture as described in Section 3.2:
class ModelConfig:
    """Model architecture configuration."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout_rate: float = 0.1

TrainingConfig
Training configuration parameters from Section 3.3:
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 10

Utilities
Core utility functions for data processing and evaluation.
Metrics
def compute_cer(predictions: str, targets: str) -> float:
    """Compute Character Error Rate."""

def compute_sps(source_embeddings: Tensor, target_embeddings: Tensor) -> float:
    """Compute Semantic Preservation Score."""

