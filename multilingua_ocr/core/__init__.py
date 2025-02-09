# multilingua_ocr/core/__init__.py

from .config import ModelConfig, TrainingConfig, PreprocessConfig
from .utils import (
    setup_logging,
    compute_cer,
    compute_sps,
    compute_cca,
    create_script_mask,
    load_cultural_ontology
)

__version__ = "1.0.0"
__author__ = "Mossab Ibrahim"
__email__ = "ventura@cs.byu.edu"
