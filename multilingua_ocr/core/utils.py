import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor

def setup_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """Set up logging configuration for MultiLingua-OCR."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path) if log_path else logging.NullHandler()
        ]
    )

def compute_cer(predictions: str, targets: str) -> float:
    """
    Compute Character Error Rate as defined in Section 5.1.
    
    Args:
        predictions: Predicted text
        targets: Ground truth text
    
    Returns:
        Character Error Rate score
    """
    if not targets:
        return float('inf')
    
    distances = np.zeros((len(predictions) + 1, len(targets) + 1))
    distances[0] = np.arange(len(targets) + 1)
    distances[:, 0] = np.arange(len(predictions) + 1)
    
    for i in range(1, len(predictions) + 1):
        for j in range(1, len(targets) + 1):
            if predictions[i-1] == targets[j-1]:
                distances[i][j] = distances[i-1][j-1]
            else:
                distances[i][j] = min(
                    distances[i-1][j] + 1,  # deletion
                    distances[i][j-1] + 1,  # insertion
                    distances[i-1][j-1] + 1  # substitution
                )
    
    return float(distances[-1][-1]) / len(targets)

def compute_sps(source_embeddings: Tensor, target_embeddings: Tensor) -> float:
    """
    Compute Semantic Preservation Score as defined in Section 5.1.
    
    Args:
        source_embeddings: Source language embeddings
        target_embeddings: Target language embeddings
    
    Returns:
        Semantic Preservation Score
    """
    cosine_sim = torch.nn.functional.cosine_similarity(
        source_embeddings, target_embeddings, dim=-1
    )
    return float(cosine_sim.mean())

def compute_cca(predictions: List[str], ground_truth: List[str], 
                cultural_terms: Dict[str, List[str]]) -> float:
    """
    Compute Cultural Concept Accuracy as defined in Section 5.1.
    
    Args:
        predictions: List of predicted texts
        ground_truth: List of ground truth texts
        cultural_terms: Dictionary mapping concepts to their variations
    
    Returns:
        Cultural Concept Accuracy score
    """
    total_terms = 0
    correct_terms = 0
    
    for pred, truth in zip(predictions, ground_truth):
        for concept, variations in cultural_terms.items():
            # Check ground truth for cultural terms
            truth_contains = any(term in truth for term in variations)
            if truth_contains:
                total_terms += 1
                # Check if prediction preserves the cultural term
                if any(term in pred for term in variations):
                    correct_terms += 1
    
    return correct_terms / total_terms if total_terms > 0 else 0.0

def create_script_mask(tokens: List[str], script_map: Dict[str, str]) -> Tensor:
    """
    Create script compatibility mask for attention mechanism.
    
    Args:
        tokens: List of input tokens
        script_map: Mapping of tokens to their scripts
    
    Returns:
        Binary mask tensor for attention weights
    """
    n = len(tokens)
    mask = torch.ones(n, n)
    
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            if script_map[token_i] != script_map[token_j]:
                mask[i, j] = 0
    
    return mask

def load_cultural_ontology(path: str) -> Dict[str, Dict]:
    """
    Load cultural ontology mapping for semantic processing.
    
    Args:
        path: Path to ontology file
    
    Returns:
        Dictionary containing cultural concept mappings
    """
    with open(path, 'r', encoding='utf-8') as f:
        ontology = yaml.safe_load(f)
    
    # Validate ontology structure
    required_categories = {'kinship', 'cosmology', 'ecological_knowledge', 
                         'ritual_practices', 'material_culture'}
    
    if not all(category in ontology for category in required_categories):
        raise ValueError(f"Ontology must contain all required categories: {required_categories}")
    
    return ontology
