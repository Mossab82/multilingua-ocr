import argparse
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image

from multilingua_ocr.data import (
    DocumentPreprocessor,
    DegradationAnalyzer,
    ScriptDetector
)
from multilingua_ocr.core import PreprocessConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess documents for MultiLingua-OCR')
    parser.add_argument('--input-dir', type=str, required=True,
                      help='Directory containing raw documents')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Directory to save processed documents')
    parser.add_argument('--config', type=str, required=True,
                      help='Preprocessing configuration file')
    return parser.parse_args()

def setup_logging(output_dir: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'preprocessing.log')
        ]
    )

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    preprocess_config = PreprocessConfig(**config['preprocessing'])
    
    # Initialize processors
    document_processor = DocumentPreprocessor(
        target_size=preprocess_config.target_size,
        analyze_degradation=preprocess_config.analyze_degradation,
        remove_background=preprocess_config.remove_background,
        normalize_contrast=preprocess_config.normalize_contrast
    )
    
    degradation_analyzer = DegradationAnalyzer()
    script_detector = ScriptDetector()
    
    # Process documents
    document_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    logger.info(f"Found {len(document_files)} documents to process")
    
    preprocessing_stats = {
        'total': len(document_files),
        'processed': 0,
        'failed': 0,
        'degraded': 0
    }
    
    for doc_path in tqdm(document_files, desc="Processing documents"):
        try:
            # Load image
            image = Image.open(doc_path).convert('RGB')
            
            # Process document
            processed_image, degradation_map = document_processor(image)
            
            # Save processed image
            output_path = output_dir / doc_path.relative_to(input_dir)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            processed_image.save(output_path)
            
            # Save degradation map
            if degradation_map is not None:
                degradation_path = output_path.with_stem(f"{output_path.stem}_degradation")
                cv2.imwrite(
                    str(degradation_path),
                    (degradation_map * 255).astype(np.uint8)
                )
            
            preprocessing_stats['processed'] += 1
            if degradation_map is not None and degradation_map.mean() > 0.3:
                preprocessing_stats['degraded'] += 1
                
        except Exception as e:
            logger.error(f"Failed to process {doc_path}: {str(e)}")
            preprocessing_stats['failed'] += 1
    
    # Save preprocessing statistics
    with open(output_dir / 'preprocessing_stats.yaml', 'w') as f:
        yaml.dump(preprocessing_stats, f)
    
    logger.info("Preprocessing completed. Statistics:")
    logger.info(f"Total documents: {preprocessing_stats['total']}")
    logger.info(f"Successfully processed: {preprocessing_stats['processed']}")
    logger.info(f"Failed: {preprocessing_stats['failed']}")
    logger.info(f"Degraded documents: {preprocessing_stats['degraded']}")

if __name__ == '__main__':
    main()
