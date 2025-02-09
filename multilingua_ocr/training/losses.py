import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class OCRLoss(nn.Module):
    """
    OCR loss component as described in Section 3.2.
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute OCR loss.
        
        Args:
            logits: Predicted logits [batch_size, seq_len, vocab_size]
            targets: Target indices [batch_size, seq_len]
        """
        return self.criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

class CulturalPreservationLoss(nn.Module):
    """
    Cultural preservation loss component as described in Section 3.2.
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cultural preservation loss.
        
        Args:
            pred_embeddings: Predicted cultural embeddings
            target_embeddings: Target cultural embeddings
        """
        cos_sim = F.cosine_similarity(
            pred_embeddings,
            target_embeddings,
            dim=-1
        )
        loss = 1 - cos_sim
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class ScriptAwareLoss(nn.Module):
    """
    Script-aware attention loss component as described in Section 3.2.
    """
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        attention_weights: torch.Tensor,
        script_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute script-aware attention loss.
        
        Args:
            attention_weights: Attention weight matrix
            script_mask: Script compatibility mask
        """
        # Penalize attention between incompatible scripts
        invalid_attention = attention_weights * (1 - script_mask)
        return invalid_attention.sum() / (attention_weights.size(0) * attention_weights.size(1))

class CombinedLoss(nn.Module):
    """
    Combined loss function as described in Section 3.2.
    """
    def __init__(
        self,
        ocr_weight: float = 1.0,
        cultural_weight: float = 0.5,
        script_weight: float = 0.3
    ):
        super().__init__()
        self.ocr_weight = ocr_weight
        self.cultural_weight = cultural_weight
        self.script_weight = script_weight
        
        self.ocr_loss = OCRLoss()
        self.cultural_loss = CulturalPreservationLoss()
        self.script_loss = ScriptAwareLoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Target tokens
            script_mask: Optional script compatibility mask
        """
        # OCR loss
        ocr_loss = self.ocr_loss(outputs['logits'], targets)
        
        # Cultural preservation loss
        cultural_loss = self.cultural_loss(
            outputs['cultural_embeddings'],
            outputs['target_embeddings']
        )
        
        # Script-aware attention loss
        script_loss = torch.tensor(0.0, device=ocr_loss.device)
        if script_mask is not None and 'attention_weights' in outputs:
            script_loss = self.script_loss(
                outputs['attention_weights'],
                script_mask
            )
        
        # Combine losses
        total_loss = (
            self.ocr_weight * ocr_loss +
            self.cultural_weight * cultural_loss +
            self.script_weight * script_loss
        )
        
        return total_loss
