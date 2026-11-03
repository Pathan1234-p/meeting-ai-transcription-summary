"""
Global configuration and environment loader for the Meeting AI pipeline.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "evaluation" / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_CACHE_DIR = PROJECT_ROOT / ".cache"

# Load .env file
load_dotenv(PROJECT_ROOT / ".env")


class PipelineConfig(BaseModel):
    """Pipeline configuration settings."""
    # ASR settings
    whisper_model_size: str = Field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small")
    )
    whisper_device: str = Field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu")
    )
    whisper_compute_type: str = Field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )
    vad_filter: bool = True
    beam_size: int = 5

    # Diarization settings
    hf_token: str = Field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )
    diarization_fallback: bool = True

    # Summarization settings
    summarizer_backend: str = Field(
        default_factory=lambda: os.getenv("SUMMARIZER_BACKEND", "local")
    )
    summarizer_model_name: str = Field(
        default_factory=lambda: os.getenv("SUMMARIZER_MODEL_NAME", "sshleifer/distilbart-cnn-12-6")
    )
    extractive_ratio: float = 0.15
    max_summary_length: int = 250
    min_summary_length: int = 50

    # NLI Factuality / Hallucination guard
    nli_model_name: str = Field(
        default_factory=lambda: os.getenv("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")
    )
    hallucination_threshold: float = 0.60

    # Reproducibility
    random_seed: int = Field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# Default instance
config = PipelineConfig()
