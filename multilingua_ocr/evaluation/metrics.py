import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import confusion_matrix, accuracy_score
import Levenshtein

class OCRMetrics:
    """
    Implements OCR evaluation metrics as described in Section 5.1.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_chars = 0
        self.total_errors = 0
        self.total_words = 0
        self.correct_words = 0
    
    def update(self, predictions: List[str], targets: List[str]):
        """Update metrics with new predictions."""
        for pred, target in zip(predictions, targets):
            # Character Error Rate
            distance = Levenshtein.distance(pred, target)
            self.total_errors += distance
            self.total_chars += len(target)
            
            # Word Accuracy
            pred_words = pred.split()
            target_words = target.split()
            self.total_words += len(target_words)
            self.correct_words += sum(
                p == t for p, t in zip(pred_words, target_words)
            )
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            'cer': self.total_errors / max(1, self.total_chars),
            'word_accuracy': self.correct_words / max(1, self.total_words)
        }

class CulturalPreservationMetrics:
    """
    Implements cultural preservation metrics as described in Section 5.1.
    """
    def __init__(self, ontology: Dict[str, List[str]]):
        """
        Args:
            ontology: Cultural concept ontology mapping
        """
        self.ontology = ontology
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_concepts = 0
        self.preserved_concepts = 0
        self.semantic_scores = []
        self.concept_matrix = None
    
    def update(
        self,
        predictions: List[str],
        targets: List[str],
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ):
        """Update metrics with new predictions."""
        # Cultural Concept Accuracy (CCA)
        for pred, target in zip(predictions, targets):
            concepts_found = 0
            concepts_preserved = 0
            
            for concept_set in self.ontology.values():
                for concept in concept_set:
                    if concept in target:
                        concepts_found += 1
                        if concept in pred:
                            concepts_preserved += 1
            
            self.total_concepts += concepts_found
            self.preserved_concepts += concepts_preserved
        
        # Semantic Preservation Score (SPS)
        sps = torch.nn.functional.cosine_similarity(
            pred_embeddings, target_embeddings, dim=-1
        ).mean().item()
        self.semantic_scores.append(sps)
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        return {
            'cca': self.preserved_concepts / max(1, self.total_concepts),
            'sps': np.mean(self.semantic_scores),
            'sps_std': np.std(self.semantic_scores)
        }

class ScriptAccuracyMetrics:
    """
    Implements script detection and classification metrics as described in Section 5.2.
    """
    def __init__(self, scripts: List[str]):
        """
        Args:
            scripts: List of script names (e.g., ['Spanish', 'Nahuatl'])
        """
        self.scripts = scripts
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.y_true = []
        self.y_pred = []
    
    def update(
        self,
        predictions: List[str],
        targets: List[str]
    ):
        """Update metrics with new predictions."""
        self.y_true.extend(targets)
        self.y_pred.extend(predictions)
    
    def compute(self) -> Dict[str, Union[float, np.ndarray]]:
        """Compute final metrics."""
        conf_matrix = confusion_matrix(
            self.y_true,
            self.y_pred,
            labels=self.scripts
        )
        
        accuracy = accuracy_score(self.y_true, self.y_pred)
        
        # Per-script accuracy
        per_script = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        
        return {
            'accuracy': accuracy,
            'per_script_accuracy': dict(zip(self.scripts, per_script)),
            'confusion_matrix': conf_matrix
        }
