import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from multilingua_ocr.data import DocumentDataset
from multilingua_ocr.models import DocumentEncoder, MultilingualDecoder
from multilingua_ocr.training import MultiLinguaTrainer, CombinedLoss
from multilingua_ocr.core import ModelConfig, TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Train MultiLingua-OCR model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Path to save outputs')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    return parser.parse_args()

def setup_logging(output_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'training.log')
        ]
    )

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    model_config = ModelConfig(**config['model'])
    training_config = TrainingConfig(**config['training'])
    
    # Initialize datasets
    train_dataset = DocumentDataset(
        data_root=args.data_dir,
        language_pairs=model_config.language_pairs,
        split='train'
    )
    val_dataset = DocumentDataset(
        data_root=args.data_dir,
        language_pairs=model_config.language_pairs,
        split='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    encoder = DocumentEncoder(**model_config.__dict__)
    decoder = MultilingualDecoder(**model_config.__dict__)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW([
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ], lr=training_config.learning_rate)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config.num_epochs
    )
    
    # Initialize loss function
    loss_fn = CombinedLoss(
        ocr_weight=training_config.ocr_loss_weight,
        cultural_weight=training_config.cultural_loss_weight,
        script_weight=training_config.script_loss_weight
    )
    
    # Initialize trainer
    trainer = MultiLinguaTrainer(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=config,
        device=args.device
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        num_epochs=training_config.num_epochs,
        early_stopping_patience=training_config.early_stopping_patience
    )
    
    # Save final model
    trainer.save_checkpoint(output_dir / 'final_model.pth')
    
    # Save training history
    with open(output_dir / 'history.yaml', 'w') as f:
        yaml.dump(history, f)

if __name__ == '__main__':
    main()
