Models API Reference
Document Encoder
Implementation of the encoder architecture from Section 3.2.
class DocumentEncoder(nn.Module):
    """Document encoder with degradation-aware features."""
    
    def forward(
        self,
        images: torch.Tensor,
        degradation_maps: Optional[torch.Tensor] = None,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder."""

Multilingual Decoder
class MultilingualDecoder(nn.Module):
    """Decoder with cultural-semantic integration."""
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        script_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through decoder."""

Attention Mechanism
class ScriptAwareAttention(nn.Module):
    """Script-aware attention mechanism."""
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        script_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute attention with script awareness."""

