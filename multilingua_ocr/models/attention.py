import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ScriptAwareAttention(nn.Module):
    """
    Script-aware attention mechanism as described in Section 3.2.
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with script-aware attention.
        
        Args:
            query, key, value: Input tensors [batch_size, seq_len, hidden_size]
            mask: Optional attention mask
            script_mask: Optional script compatibility mask
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(key).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(value).view(
            batch_size, -1, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply script compatibility mask
        if script_mask is not None:
            scores = scores * script_mask.unsqueeze(1)
        
        # Apply attention mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(
            batch_size, -1, self.hidden_size
        )
        
        return self.out_proj(out)
