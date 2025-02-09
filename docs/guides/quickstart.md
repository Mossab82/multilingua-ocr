Quick Start Guide
Basic Usage
1. Prepare Data
from multilingua_ocr.data import DocumentPreprocessor

# Initialize preprocessor
processor = DocumentPreprocessor()

# Process document
processed_image, degradation_map = processor(image)

2. Load Model
from multilingua_ocr.models import DocumentEncoder, MultilingualDecoder

# Initialize model
encoder = DocumentEncoder()
decoder = MultilingualDecoder()

# Load pre-trained weights
checkpoint = torch.load('models/checkpoints/best_model.pth')
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])

3. Process Documents
# Process single document
encoded = encoder(image, degradation_map)
outputs = decoder.generate(encoded)

