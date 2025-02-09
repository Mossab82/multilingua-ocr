#!/bin/bash

# Exit on error
set -e

# Default values
CONFIG="configs/default_config.yaml"
DATA_DIR="data/processed"
OUTPUT_DIR="models/checkpoints"
NUM_GPUS=1

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Activate virtual environment
source venv/bin/activate

echo "Starting training with configuration:"
echo "Config file: $CONFIG"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"

# If multiple GPUs, use distributed training
if [ $NUM_GPUS -gt 1 ]; then
    echo "Using distributed training with $NUM_GPUS GPUs"
    python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS \
        multilingua_ocr/scripts/train.py \
        --config $CONFIG \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR \
        --distributed
else
    echo "Using single GPU/CPU training"
    python multilingua_ocr/scripts/train.py \
        --config $CONFIG \
        --data-dir $DATA_DIR \
        --output-dir $OUTPUT_DIR
fi

# Save training configuration
cp $CONFIG $OUTPUT_DIR/

echo "Training completed successfully!"
