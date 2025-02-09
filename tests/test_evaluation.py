import pytest
import torch
import numpy as np
from multilingua_ocr.evaluation import (
    OCRMetrics,
    CulturalPreservationMetrics,
    ScriptAccuracyMetrics
)

@pytest.fixture
def sample_predictions():
    return [
        "El tlatoani mandó preparar",
        "Los tepaneca llegaron"
    ]

@pytest.fixture
def sample_targets():
    return [
        "El tlatoani mandó preparar",
        "Los tepanecah llegaron"
    ]

@pytest.fixture
def sample_ontology():
    return {
        'titles': ['tlatoani', 'tepaneca'],
        'actions': ['mandó', 'llegaron'],
    }

def test_ocr_metrics(sample_predictions, sample_targets):
    metrics = OCRMetrics()
    metrics.update(sample_predictions, sample_targets)
    results = metrics.compute()
    
    assert 'cer' in results
    assert 'word_accuracy' in results
    assert 0 <= results['cer'] <= 1
    assert 0 <= results['word_accuracy'] <= 1

def test_cultural_preservation_metrics(
    sample_predictions,
    sample_targets,
    sample_ontology
):
    metrics = CulturalPreservationMetrics(sample_ontology)
    
    pred_embeddings = torch.randn(2, 256)
    target_embeddings = torch.randn(2, 256)
    
    metrics.update(
        sample_predictions,
        sample_targets,
        pred_embeddings,
        target_embeddings
    )
    results = metrics.compute()
    
    assert 'cca' in results
    assert 'sps' in results
    assert 0 <= results['cca'] <= 1
    assert 0 <= results['sps'] <= 1

def test_script_accuracy_metrics():
    metrics = ScriptAccuracyMetrics(
        scripts=['Spanish', 'Nahuatl']
    )
    
    predictions = ['Spanish', 'Nahuatl', 'Spanish']
    targets = ['Spanish', 'Spanish', 'Spanish']
    
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    assert 'accuracy' in results
    assert 'per_script_accuracy' in results
    assert 'confusion_matrix' in results
    assert isinstance(results['confusion_matrix'], np.ndarray)
