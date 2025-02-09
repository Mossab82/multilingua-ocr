import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import ScriptAwareAttention

class DocumentEncoder(nn.Module):
    """
    Document encoder with degradation-aware feature extraction as described in Section 3.2.
    """
    def __init__(
        self,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        max_sequence_length: int = 512
    ):
        super().__init__()
        
        # CNN backbone for visual feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU()
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, max_sequence_length, hidden_size)
        )
        
        # Degradation embedding
        self.degradation_embed = nn.Linear(1, hidden_size)
        
        # Transformer encoder layers with script-aware attention
        self.layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        images: torch.Tensor,
        degradation_maps: Optional[torch.Tensor] = None,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            images: [batch_size, channels, height, width]
            degradation_maps: Optional [batch_size, height, width]
            script_mask: Optional [batch_size, seq_len, seq_len]
            
        Returns:
            Encoded features [batch_size, seq_len, hidden_size]
        """
        # Extract visual features
        features = self.cnn(images)
        b, c, h, w = features.shape
        features = features.reshape(b, c, -1).transpose(1, 2)
        
        # Add positional encoding
        features = features + self.pos_encoding[:, :features.size(1)]
        
        # Incorporate degradation information if available
        if degradation_maps is not None:
            deg_embed = self.degradation_embed(degradation_maps.unsqueeze(-1))
            features = features + deg_embed
        
        # Apply transformer layers
        features = self.dropout(features)
        for layer in self.layers:
            features = layer(features, script_mask)
        
        return self.norm(features)

class EncoderLayer(nn.Module):
    """Single encoder layer with script-aware attention."""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = ScriptAwareAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
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
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with script awareness
        attn_out = self.attention(x, x, x, script_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
