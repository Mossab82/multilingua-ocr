Model Training Guide
Training Pipeline
1. Data Preparation
from multilingua_ocr.data import DocumentDataset
from torch.utils.data import DataLoader

# Create datasets
train_dataset = DocumentDataset(data_root="data/train")
val_dataset = DocumentDataset(data_root="data/val")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)
2. Model Configuration
from multilingua_ocr.core import ModelConfig, TrainingConfig

model_config = ModelConfig()
training_config = TrainingConfig()
3. Training Loop
from multilingua_ocr.training import MultiLinguaTrainer

trainer = MultiLinguaTrainer(
    encoder=encoder,
    decoder=decoder,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)

trainer.train(num_epochs=100)
