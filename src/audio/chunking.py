"""
Audio chunking module for long meeting recordings.
Splits large audio files into overlapping segments and provides offset alignment.
"""

import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def chunk_audio(
    audio_path: str,
    chunk_length_sec: float = 600.0,  # 10 minutes
    overlap_sec: float = 2.0,        # 2 seconds
    output_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Split audio into overlapping chunks for memory-safe and robust ASR processing.

    Args:
        audio_path: Path to preprocessed 16kHz WAV file.
        chunk_length_sec: Target duration of each chunk in seconds.
        overlap_sec: Overlap between consecutive chunks in seconds.
        output_dir: Directory where audio chunks are saved.

    Returns:
        List of dictionaries with:
        - 'chunk_index': int
        - 'chunk_path': str
        - 'start_sec': float
        - 'end_sec': float
        - 'offset_sec': float
    """
    audio_file = Path(audio_path).resolve()
    if output_dir is None:
        chunks_dir = audio_file.parent / f"{audio_file.stem}_chunks"
    else:
        chunks_dir = Path(output_dir).resolve()

    chunks_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(str(audio_file))
    total_duration_sec = len(audio) / 1000.0

    # If the audio duration is shorter than or equal to chunk_length_sec + overlap_sec, do not split
    if total_duration_sec <= chunk_length_sec + overlap_sec:
        return [{
            "chunk_index": 0,
            "chunk_path": str(audio_file),
            "start_sec": 0.0,
            "end_sec": total_duration_sec,
            "offset_sec": 0.0,
        }]

    chunk_length_ms = int(chunk_length_sec * 1000)
    overlap_ms = int(overlap_sec * 1000)
    step_ms = chunk_length_ms - overlap_ms

    chunks_info = []
    current_start_ms = 0
    chunk_idx = 0

    while current_start_ms < len(audio):
        current_end_ms = min(current_start_ms + chunk_length_ms, len(audio))
        chunk_audio_segment = audio[current_start_ms:current_end_ms]

        chunk_path = chunks_dir / f"chunk_{chunk_idx:03d}_{current_start_ms//1000}s.wav"
        chunk_audio_segment.export(str(chunk_path), format="wav")

        chunks_info.append({
            "chunk_index": chunk_idx,
            "chunk_path": str(chunk_path),
            "start_sec": current_start_ms / 1000.0,
            "end_sec": current_end_ms / 1000.0,
            "offset_sec": current_start_ms / 1000.0,
        })

        if current_end_ms >= len(audio):
            break

        current_start_ms += step_ms
        chunk_idx += 1

    logger.info(f"Split {total_duration_sec:.1f}s audio into {len(chunks_info)} chunks.")
    return chunks_info
