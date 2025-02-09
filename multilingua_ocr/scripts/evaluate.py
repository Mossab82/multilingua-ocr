import argparse
import logging
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import json

from multilingua_ocr.data import DocumentDataset
from multilingua_ocr.models import DocumentEncoder, MultilingualDecoder
from multilingua_ocr.evaluation import (
    OCRMetrics,
    CulturalPreservationMetrics,
    ScriptAccuracyMetrics,
    ResultsPlotter
)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MultiLingua-OCR model')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                      help='Path to test dataset directory')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Path to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for evaluation')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for evaluation')
    return parser.parse_args()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=args.device)
    config = checkpoint['config']
    
    # Initialize model
    encoder = DocumentEncoder(**config['model'])
    decoder = MultilingualDecoder(**config['model'])
    
    # Load model weights
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.to(args.device)
    decoder.to(args.device)
    encoder.eval()
    decoder.eval()
    
    # Initialize dataset
    test_dataset = DocumentDataset(
        data_root=args.data_dir,
        language_pairs=config['model']['language_pairs'],
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize metrics
    ocr_metrics = OCRMetrics()
    cultural_metrics = CulturalPreservationMetrics(
        ontology=config['cultural_ontology']
    )
    script_metrics = ScriptAccuracyMetrics(
        scripts=config['model']['language_pairs']
    )
    
    # Evaluate
    results = {}
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            images = batch['images'].to(args.device)
            script_masks = batch['script_masks'].to(args.device)
            degradation_maps = batch['degradation_maps'].to(args.device)
            
            # Forward pass
            encoded = encoder(
                images,
                degradation_maps=degradation_maps,
                script_mask=script_masks
            )
            outputs = decoder.generate(encoded, script_mask=script_masks)
            
            # Update metrics
            ocr_metrics.update(outputs, batch['raw_texts'])
            cultural_metrics.update(
                predictions=outputs,
                targets=batch['raw_texts'],
                pred_embeddings=outputs['cultural_embeddings'],
                target_embeddings=batch['cultural_embeddings']
            )
            script_metrics.update(
                predictions=outputs['script_predictions'],
                targets=batch['script_labels']
            )
    
    # Compute final metrics
    results['ocr'] = ocr_metrics.compute()
    results['cultural'] = cultural_metrics.compute()
    results['script'] = script_metrics.compute()
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plotter = ResultsPlotter()
    
    # Plot confusion matrix
    plotter.plot_confusion_matrix(
        results['script']['confusion_matrix'],
        config['model']['language_pairs'],
        save_path=output_dir / 'confusion_matrix.png'
    )
    
    # Plot performance comparison
    plotter.plot_language_pair_comparison(
        results,
        metrics=['cer', 'cca', 'sps'],
        save_path=output_dir / 'performance_comparison.png'
    )

if __name__ == '__main__':
    main()
