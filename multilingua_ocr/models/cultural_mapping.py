import torch
import torch.nn as nn
from typing import Dict, Optional

class CulturalMapper(nn.Module):
    """
    Cultural-semantic mapping module as described in Section 3.2.
    """
    def __init__(
        self,
        hidden_size: int,
        num_concepts: int = 1000,
        concept_dim: int = 256
    ):
        super().__init__()
        
        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.concept_attention = nn.MultiheadAttention(
            hidden_size, 8, dropout=0.1
        )
        
        self.cultural_proj = nn.Sequential(
            nn.Linear(hidden_size + concept_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.concept_classifier = nn.Linear(hidden_size, num_concepts)
    
    def forward(
        self,
        x: torch.Tensor,
        concepts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cultural-semantic mapping.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            concepts: Optional concept IDs [batch_size, num_concepts]
            
        Returns:
            Dictionary containing:
                - mapped_features: Culturally-aware features
                - concept_logits: Concept classification logits
                - concept_attention: Attention weights over concepts
        """
        # Compute concept embeddings
        if concepts is None:
            # Use all concepts if none specified
            concepts = torch.arange(
                self.concept_embedding.num_embeddings,
                device=x.device
            )
        concept_embeds = self.concept_embedding(concepts)
        
        # Apply concept attention
        attn_out, attn_weights = self.concept_attention(
            x.transpose(0, 1),
       concept_embeds.transpose(0, 1),
            concept_embeds.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Concatenate attention output with input features
        combined = torch.cat([x, attn_out], dim=-1)
        
        # Project to cultural-semantic space
        mapped_features = self.cultural_proj(combined)
        
        # Classify cultural concepts
        concept_logits = self.concept_classifier(mapped_features)
        
        return {
            'mapped_features': mapped_features,
            'concept_logits': concept_logits,
            'concept_attention': attn_weights
        }

class CulturalLoss(nn.Module):
    """
    Loss function for cultural concept preservation as described in Section 3.2.
    """
    def __init__(
        self,
        concept_weight: float = 0.5,
        semantic_weight: float = 0.5
    ):
        super().__init__()
        self.concept_weight = concept_weight
        self.semantic_weight = semantic_weight
        
        self.concept_loss = nn.CrossEntropyLoss()
        self.semantic_loss = nn.CosineEmbeddingLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_concepts: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cultural preservation loss.
        
        Args:
            outputs: Dictionary from CulturalMapper
            target_concepts: Ground truth concept labels
            target_embeddings: Target semantic embeddings
            
        Returns:
            Combined loss value
        """
        # Concept classification loss
        concept_loss = self.concept_loss(
            outputs['concept_logits'].view(-1, outputs['concept_logits'].size(-1)),
            target_concepts.view(-1)
        )
        
        # Semantic preservation loss
        semantic_loss = self.semantic_loss(
            outputs['mapped_features'],
            target_embeddings,
            torch.ones(outputs['mapped_features'].size(0), device=outputs['mapped_features'].device)
        )
        
        # Combine losses
        total_loss = (
            self.concept_weight * concept_loss +
            self.semantic_weight * semantic_loss
        )
        
        return total_loss

class IndigenousLanguageAdapter(nn.Module):
    """
    Adapter module for Indigenous language features as described in Section 3.2.
    """
    def __init__(
        self,
        hidden_size: int,
        language_pair: Tuple[str, str],
        bottleneck_size: int = 64
    ):
        super().__init__()
        
        self.language_pair = language_pair
        
        # Down-projection
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        
        # Language-specific transformation
        self.language_transform = nn.Sequential(
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, bottleneck_size)
        )
        
        # Up-projection
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply language-specific adaptation.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted features [batch_size, seq_len, hidden_size]
        """
        identity = x
        
        # Down-project
        x = self.down_proj(x)
        
        # Apply language-specific transformation
        x = self.language_transform(x)
        
        # Up-project
        x = self.up_proj(x)
        
        # Add residual and normalize
        x = self.layer_norm(x + identity)
        
        return x

class OntologyIntegrator(nn.Module):
    """
    Integrates cultural ontology information as described in Section 3.2.
    """
    def __init__(
        self,
        hidden_size: int,
        ontology_size: int = 1000,
        num_relations: int = 5
    ):
        super().__init__()
        
        self.ontology_embedding = nn.Embedding(ontology_size, hidden_size)
        self.relation_embedding = nn.Embedding(num_relations, hidden_size)
        
        self.integration_layer = nn.MultiheadAttention(
            hidden_size, 8, dropout=0.1
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        ontology_ids: torch.Tensor,
        relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate ontological knowledge.
        
        Args:
            x: Input features [batch_size, seq_len, hidden_size]
            ontology_ids: Ontology concept IDs
            relation_ids: Relation type IDs
            
        Returns:
            Knowledge-enhanced features [batch_size, seq_len, hidden_size]
        """
        # Get embeddings
        onto_embeds = self.ontology_embedding(ontology_ids)
        rel_embeds = self.relation_embedding(relation_ids)
        
        # Combine ontology and relation embeddings
        knowledge = onto_embeds + rel_embeds
        
        # Apply attention
        attn_out, _ = self.integration_layer(
            x.transpose(0, 1),
            knowledge.transpose(0, 1),
            knowledge.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # Combine with input features
        combined = torch.cat([x, attn_out], dim=-1)
        output = self.output_proj(combined)
        
        return output
