"""
ASR Transcriber module wrapping faster-whisper.
Extracts word-level timestamps, handles silence filtering, and supports chunked transcription.
"""

import time
import logging
from typing import List, Optional, Tuple
import torch
from faster_whisper import WhisperModel
from src.schemas import TranscriptSegment, WordTimestamp
from src.config import config

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Wrapper around faster-whisper WhisperModel."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.model_size = model_size or config.whisper_model_size
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self.beam_size = beam_size
        self.vad_filter = vad_filter

        logger.info(
            f"Initializing WhisperModel: size={self.model_size}, device={self.device}, compute_type={self.compute_type}"
        )
        self.model = WhisperModel(
            model_size_or_path=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(
        self,
        audio_path: str,
        initial_prompt: Optional[str] = None,
        language: str = "en",
        time_offset: float = 0.0,
    ) -> Tuple[List[TranscriptSegment], float]:
        """
        Transcribe an audio file into structured segments with word-level timestamps.

        Args:
            audio_path: Path to the 16kHz WAV audio.
            initial_prompt: Optional context prompt for Whisper.
            language: Language code (default 'en').
            time_offset: Global time offset in seconds (for audio chunks).

        Returns:
            Tuple of (segments, transcription_time_sec)
        """
        start_time = time.time()
        logger.info(f"Transcribing {audio_path} (offset={time_offset:.2f}s)...")

        segments_iter, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True,
            initial_prompt=initial_prompt,
        )

        segments: List[TranscriptSegment] = []
        seg_id = 0

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            word_list: List[WordTimestamp] = []
            if seg.words:
                for w in seg.words:
                    word_list.append(
                        WordTimestamp(
                            word=w.word.strip(),
                            start=round(w.start + time_offset, 3),
                            end=round(w.end + time_offset, 3),
                            probability=round(float(w.probability), 3),
                        )
                    )

            segments.append(
                TranscriptSegment(
                    segment_id=seg_id,
                    speaker="Speaker unknown",
                    start=round(seg.start + time_offset, 3),
                    end=round(seg.end + time_offset, 3),
                    text=text,
                    raw_text=text,
                    words=word_list,
                )
            )
            seg_id += 1

        elapsed = time.time() - start_time
        logger.info(f"Transcription complete: {len(segments)} segments in {elapsed:.2f}s.")
        return segments, elapsed
