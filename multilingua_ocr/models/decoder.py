import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .attention import ScriptAwareAttention
from .cultural_mapping import CulturalMapper

class MultilingualDecoder(nn.Module):
    """
    Multilingual decoder with cultural-semantic integration as described in Section 3.2.
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_length: int = 512
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_length, hidden_size)
        )
        
        # Cultural mapper for semantic preservation
        self.cultural_mapper = CulturalMapper(hidden_size)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        script_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through decoder.
        
        Args:
            tgt: Target tokens [batch_size, seq_len]
            memory: Encoder outputs
            tgt_mask: Target sequence mask
            memory_mask: Memory sequence mask
            script_mask: Script compatibility mask
            
        Returns:
            Logits and cultural mapping outputs
        """
        x = self.embedding(tgt)
        x = x + self.pos_encoding[:, :x.size(1)]
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(
                x, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                script_mask=script_mask
            )
        
        x = self.norm(x)
        
        # Apply cultural mapping
        cultural_outputs = self.cultural_mapper(x)
        
        # Generate logits
        logits = self.output_proj(x)
        
        return logits, cultural_outputs

class DecoderLayer(nn.Module):
    """Single decoder layer with script-aware attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = ScriptAwareAttention(hidden_size, num_heads)
        self.cross_attn = ScriptAwareAttention(hidden_size, num_heads)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn))
        
        # Cross-attention with script awareness
        cross_attn = self.cross_attn(
            x, memory, memory,
            mask=memory_mask,
            script_mask=script_mask
        )
        x = self.norm2(x + self.dropout(cross_attn))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x
