import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm
from ..evaluation import OCRMetrics, CulturalPreservationMetrics
from ..models import DocumentEncoder, MultilingualDecoder
from .losses import CombinedLoss

class MultiLinguaTrainer:
    """
    Trainer class implementing the training procedure from Section 3.3.
    """
    def __init__(
        self,
        encoder: DocumentEncoder,
        decoder: MultilingualDecoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_fn: CombinedLoss,
        config: dict,
        device: str = 'cuda'
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        
        # Initialize metrics
        self.ocr_metrics = OCRMetrics()
        self.cultural_metrics = CulturalPreservationMetrics(
            ontology=config['cultural_ontology']
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'cer': [],
            'cca': [],
            'sps': []
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.encoder.train()
        self.decoder.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        with tqdm(total=num_batches, desc='Training') as pbar:
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                
                # Move data to device
                images = batch['images'].to(self.device)
                texts = batch['texts'].to(self.device)
                script_masks = batch['script_masks'].to(self.device)
                degradation_maps = batch['degradation_maps'].to(self.device)
                
                # Forward pass
                encoded = self.encoder(
                    images, 
                    degradation_maps=degradation_maps,
                    script_mask=script_masks
                )
                outputs = self.decoder(
                    texts[:, :-1],
                    memory=encoded,
                    script_mask=script_masks
                )
                
                # Compute loss
                loss = self.loss_fn(
                    outputs=outputs,
                    targets=texts[:, 1:],
                    script_mask=script_masks
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + 
                    list(self.decoder.parameters()),
                    self.config['max_grad_norm']
                )
                self.optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        return {'loss': epoch_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Perform validation."""
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0.0
        self.ocr_metrics.reset()
        self.cultural_metrics.reset()
        
        for batch in self.val_loader:
            # Move data to device
            images = batch['images'].to(self.device)
            texts = batch['texts'].to(self.device)
            script_masks = batch['script_masks'].to(self.device)
            degradation_maps = batch['degradation_maps'].to(self.device)
            
            # Forward pass
            encoded = self.encoder(
                images,
                degradation_maps=degradation_maps,
                script_mask=script_masks
            )
            outputs = self.decoder(
                texts[:, :-1],
                memory=encoded,
                script_mask=script_masks
            )
            
            # Compute loss
            loss = self.loss_fn(
                outputs=outputs,
                targets=texts[:, 1:],
                script_mask=script_masks
            )
            total_loss += loss.item()
            
            # Update metrics
            predictions = self.decoder.generate(encoded, script_mask=script_masks)
            self.ocr_metrics.update(predictions, batch['raw_texts'])
            self.cultural_metrics.update(
                predictions=predictions,
                targets=batch['raw_texts'],
                pred_embeddings=outputs['cultural_embeddings'],
                target_embeddings=batch['cultural_embeddings']
            )
        
        # Compute metrics
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            **self.ocr_metrics.compute(),
            **self.cultural_metrics.compute()
        }
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.
        
        Args:
            num_epochs: Maximum number of epochs to train
            early_stopping_patience: Number of epochs to wait for improvement
        
        Returns:
            Training history dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch + 1}/{num_epochs}')
            
            # Training
            train_metrics = self.train_epoch()
            self.history['train_loss'].append(train_metrics['loss'])
            
            # Validation
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['cer'].append(val_metrics['cer'])
            self.history['cca'].append(val_metrics['cca'])
            self.history['sps'].append(val_metrics['sps'])
            
            # Log metrics
            self.logger.info(
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["val_loss"]:.4f}, '
                f'CER: {val_metrics["cer"]:.4f}, '
                f'CCA: {val_metrics["cca"]:.4f}, '
                f'SPS: {val_metrics["sps"]:.4f}'
            )
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Early stopping
            if early_stopping_patience:
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    # Save best model
                    self.save_checkpoint('best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        self.logger.info('Early stopping triggered')
                        break
        
        return self.history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.config = checkpoint['config']
        self.history = checkpoint['history']
