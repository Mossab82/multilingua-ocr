Evaluation Guide
Evaluation Pipeline
1. Load Test Data
test_dataset = DocumentDataset(data_root="data/test")
test_loader = DataLoader(test_dataset, batch_size=32)
2. Run Evaluation
from multilingua_ocr.evaluation import (
    OCRMetrics,
    CulturalPreservationMetrics
)

# Initialize metrics
ocr_metrics = OCRMetrics()
cultural_metrics = CulturalPreservationMetrics()

# Evaluate
results = evaluate_model(
    encoder,
    decoder,
    test_loader,
    ocr_metrics,
    cultural_metrics
)
3. Generate Reports
from multilingua_ocr.evaluation import generate_report

generate_report(results, output_dir="results")
