"""Speaker diarization and timestamp alignment module."""
from .diarizer import SpeakerDiarizer, DiarizationInterval
from .merger import align_words_with_speakers, apply_speaker_mapping

__all__ = [
    "SpeakerDiarizer",
    "DiarizationInterval",
    "align_words_with_speakers",
    "apply_speaker_mapping",
]
