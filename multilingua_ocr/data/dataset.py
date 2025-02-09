from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image

class MultilingualBatch:
    """Batch container for multilingual document processing."""
    def __init__(
        self,
        images: torch.Tensor,
        texts: List[str],
        script_masks: torch.Tensor,
        language_pairs: List[Tuple[str, str]],
        degradation_maps: Optional[torch.Tensor] = None
    ):
        self.images = images  # [B, C, H, W]
        self.texts = texts
        self.script_masks = script_masks  # [B, seq_len, seq_len]
        self.language_pairs = language_pairs
        self.degradation_maps = degradation_maps  # [B, H, W] if provided

class DocumentDataset(Dataset):
    """Dataset for multilingual historical documents as described in Section 4.2."""
    
    def __init__(
        self,
        data_root: str,
        language_pairs: List[Tuple[str, str]],
        split: str = 'train',
        max_length: int = 512,
        preprocessor: Optional['DocumentPreprocessor'] = None
    ):
        """
        Args:
            data_root: Root directory of dataset
            language_pairs: List of (source, target) language pairs
            split: Data split ('train', 'val', 'test')
            max_length: Maximum sequence length
            preprocessor: Optional document preprocessor
        """
        self.data_root = Path(data_root)
        self.language_pairs = language_pairs
        self.max_length = max_length
        self.preprocessor = preprocessor or DocumentPreprocessor()
        
        # Load document metadata
        self.metadata = pd.read_csv(self.data_root / f'{split}_metadata.csv')
        
        # Initialize script detector
        self.script_detector = ScriptDetector()
        
        # Load cultural ontology
        self.ontology = self._load_ontology()
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a document sample with its annotations."""
        row = self.metadata.iloc[idx]
        
        # Load and preprocess image
        image_path = self.data_root / row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.preprocessor:
            image, degradation_map = self.preprocessor(image)
        
        # Load text and language information
        source_text = row['source_text']
        target_text = row['target_text']
        source_lang = row['source_language']
        target_lang = row['target_language']
        
        # Create script mask
        script_mask = self.script_detector.create_mask(
            source_text + target_text,
            (source_lang, target_lang)
        )
        
        return {
            'image': image,
            'source_text': source_text,
            'target_text': target_text,
            'script_mask': script_mask,
            'degradation_map': degradation_map,
            'language_pair': (source_lang, target_lang)
        }
    
    def _load_ontology(self) -> Dict:
        """Load cultural concept ontology."""
        ontology_path = self.data_root / 'cultural_ontology.yaml'
        return load_cultural_ontology(ontology_path)
