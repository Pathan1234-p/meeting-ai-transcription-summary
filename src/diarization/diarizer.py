"""
Speaker diarization module wrapping pyannote.audio with graceful fallback.
"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional
from src.config import config

logger = logging.getLogger(__name__)


@dataclass
class DiarizationInterval:
    """Temporal interval attributed to a speaker."""
    speaker: str
    start: float
    end: float


class SpeakerDiarizer:
    """
    Performs speaker diarization using pyannote.audio if available and authenticated,
    otherwise falls back gracefully to default speaker attribution without breaking.
    """

    def __init__(self, hf_token: Optional[str] = None):
        self.hf_token = hf_token or config.hf_token
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        """Attempt to load pyannote.audio speaker-diarization pipeline."""
        if not self.hf_token:
            logger.warning(
                "No Hugging Face token provided in HF_TOKEN. "
                "Speaker diarization will gracefully fall back to 'Speaker unknown'."
            )
            return

        try:
            import torch
            from pyannote.audio import Pipeline

            logger.info("Initializing pyannote/speaker-diarization-3.1 pipeline...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline.to(torch.device(device))
            logger.info(f"pyannote.audio loaded successfully on {device}.")
        except ImportError:
            logger.warning(
                "pyannote.audio is not installed. Diarization will fall back gracefully to 'Speaker unknown'."
            )
        except Exception as e:
            logger.warning(
                f"Could not initialize pyannote.audio pipeline ({e}). "
                "Falling back gracefully to 'Speaker unknown'."
            )
            self.pipeline = None

    def diarize(self, audio_path: str) -> List[DiarizationInterval]:
        """
        Diarize audio file into speaker intervals.

        Args:
            audio_path: Path to 16kHz mono WAV file.

        Returns:
            List of DiarizationInterval instances.
        """
        if self.pipeline is None:
            logger.info("Diarization fallback active: using single/unknown speaker.")
            return []

        try:
            logger.info(f"Running pyannote diarization on {audio_path}...")
            diarization = self.pipeline(audio_path)
            intervals: List[DiarizationInterval] = []

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                intervals.append(
                    DiarizationInterval(
                        speaker=speaker,
                        start=round(turn.start, 3),
                        end=round(turn.end, 3),
                    )
                )

            logger.info(f"Diarization extracted {len(intervals)} speaker turns.")
            return intervals
        except Exception as e:
            logger.warning(f"Error during pyannote diarization run ({e}). Falling back gracefully.")
            return []
