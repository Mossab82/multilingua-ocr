import pytest
import torch
import numpy as np
from multilingua_ocr.models import (
    DocumentEncoder,
    MultilingualDecoder,
    ScriptAwareAttention,
    CulturalMapper
)
from multilingua_ocr.core import ModelConfig

@pytest.fixture
def model_config():
    return ModelConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout_rate=0.1,
        max_sequence_length=128
    )

@pytest.fixture
def sample_batch():
    batch_size = 2
    height, width = 224, 224
    seq_len = 64
    
    return {
        'images': torch.randn(batch_size, 3, height, width),
        'texts': torch.randint(0, 1000, (batch_size, seq_len)),
        'script_masks': torch.ones(batch_size, seq_len, seq_len),
        'degradation_maps': torch.rand(batch_size, height, width)
    }

def test_encoder_forward(model_config, sample_batch):
    encoder = DocumentEncoder(**model_config.__dict__)
    encoder.eval()
    
    with torch.no_grad():
        output = encoder(
            sample_batch['images'],
            degradation_maps=sample_batch['degradation_maps'],
            script_mask=sample_batch['script_masks']
        )
    
    expected_shape = (
        sample_batch['images'].size(0),
        sample_batch['texts'].size(1),
        model_config.hidden_size
    )
    assert output.shape == expected_shape

def test_decoder_forward(model_config, sample_batch):
    decoder = MultilingualDecoder(
        vocab_size=1000,
        **model_config.__dict__
    )
    decoder.eval()
    
    memory = torch.randn(
        sample_batch['images'].size(0),
        sample_batch['texts'].size(1),
        model_config.hidden_size
    )
    
    with torch.no_grad():
        logits, cultural_outputs = decoder(
            sample_batch['texts'],
            memory,
            script_mask=sample_batch['script_masks']
        )
    
    assert logits.shape[:-1] == sample_batch['texts'].shape
    assert logits.shape[-1] == 1000  # vocab_size

def test_script_aware_attention():
    attention = ScriptAwareAttention(
        hidden_size=256,
        num_heads=4
    )
    batch_size = 2
    seq_len = 64
    
    query = torch.randn(batch_size, seq_len, 256)
    script_mask = torch.ones(batch_size, seq_len, seq_len)
    
    output = attention(query, query, query, script_mask=script_mask)
    assert output.shape == query.shape

def test_cultural_mapper():
    mapper = CulturalMapper(
        hidden_size=256,
        num_concepts=100,
        concept_dim=64
    )
    batch_size = 2
    seq_len = 64
    
    features = torch.randn(batch_size, seq_len, 256)
    outputs = mapper(features)
    
    assert 'mapped_features' in outputs
    assert 'concept_logits' in outputs
    assert outputs['mapped_features'].shape == features.shape
