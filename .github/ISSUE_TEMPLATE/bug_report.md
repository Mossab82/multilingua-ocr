name: Bug report
about: Report issues with MultiLingua-OCR's document processing or cultural preservation
title: '[BUG] '
labels: 'bug'
assignees: ''

Document Processing Issue
Describe the specific OCR or document processing issue encountered.
Document Details

Language Pair: [e.g., Spanish-Nahuatl, Spanish-Mixtec, Spanish-Zapotec]
Document Type: [e.g., Colonial manuscript, Administrative record, Religious text]
Document Condition: [Specify degradation level according to our metrics]
Page Count: [Number of pages processed]

# Include your configuration settings
script_aware_attention: true
cultural_mapping_enabled: true
degradation_analysis:
  enabled: true
  sensitivity: 0.8
batch_size: 32
learning_rate: 1e-4

# Include any error messages or unexpected output

Performance Metrics

Character Error Rate (CER): [%]
Semantic Preservation Score (SPS): [value]
Cultural Concept Accuracy (CCA): [%]

Environment

OS: [e.g., Ubuntu 20.04]
GPU: [e.g., Tesla V100]
CUDA Version: [e.g., 11.4]
Python Version: [e.g., 3.9]
Framework Version: [e.g., PyTorch 2.0]

Screenshots
Attach relevant document images and OCR output comparisons.
