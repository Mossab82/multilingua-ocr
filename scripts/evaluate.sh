#!/bin/bash

# Exit on error
set -e

# Default values
MODEL_PATH="models/checkpoints/best_model.pth"
TEST_DATA="data/processed/test"
OUTPUT_DIR="results/evaluation"
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test-data)
            TEST_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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

echo "Starting evaluation:"
echo "Model path: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Output directory: $OUTPUT_DIR"

# Run evaluation script
python multilingua_ocr/scripts/evaluate.py \
    --model-path $MODEL_PATH \
    --data-dir $TEST_DATA \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE

# Generate visualizations
echo "Generating result visualizations..."
python multilingua_ocr/scripts/visualize_results.py \
    --results-dir $OUTPUT_DIR \
    --output-dir $OUTPUT_DIR/figures

# Generate evaluation report
echo "Generating evaluation report..."
python -c "
from multilingua_ocr.evaluation import generate_report
generate_report('$OUTPUT_DIR')
"

echo "Evaluation completed successfully!"
echo "Results saved to: $OUTPUT_DIR"

# Make scripts executable
chmod +x scripts/setup.sh
chmod +x scripts/train.sh
chmod +x scripts/evaluate.sh
