"""
Alignment and merger module for aligning ASR word timestamps with speaker intervals.
"""

import logging
from typing import List, Dict, Optional
from src.schemas import TranscriptSegment, WordTimestamp
from .diarizer import DiarizationInterval

logger = logging.getLogger(__name__)


def find_speaker_for_word(
    word: WordTimestamp,
    intervals: List[DiarizationInterval],
    default_speaker: str = "Speaker unknown",
) -> str:
    """Find the best-matching speaker for a word based on timestamp overlap."""
    if not intervals:
        return default_speaker

    word_mid = (word.start + word.end) / 2.0

    # 1. Exact interval containment
    for interval in intervals:
        if interval.start <= word_mid <= interval.end:
            return interval.speaker

    # 2. Maximum overlap calculation
    best_speaker = default_speaker
    max_overlap = 0.0

    for interval in intervals:
        overlap_start = max(word.start, interval.start)
        overlap_end = min(word.end, interval.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = interval.speaker

    if max_overlap > 0.0:
        return best_speaker

    # 3. Nearest interval fallback
    nearest_dist = float("inf")
    nearest_speaker = default_speaker
    for interval in intervals:
        dist = min(abs(word_mid - interval.start), abs(word_mid - interval.end))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_speaker = interval.speaker

    return nearest_speaker


def align_words_with_speakers(
    segments: List[TranscriptSegment],
    intervals: List[DiarizationInterval],
    speaker_mapping: Optional[Dict[str, str]] = None,
    max_pause_split_sec: float = 2.0,
) -> List[TranscriptSegment]:
    """
    Align word-level timestamps with speaker intervals and re-chunk into
    speaker-attributed transcript segments.

    Args:
        segments: Raw transcript segments from ASR with word timestamps.
        intervals: Speaker intervals from diarization.
        speaker_mapping: Optional dictionary mapping raw speaker IDs to human names.
        max_pause_split_sec: Maximum silence between words before starting a new segment.

    Returns:
        List of aligned TranscriptSegment objects.
    """
    if not intervals:
        logger.info("No diarization intervals available. Retaining original ASR segments.")
        if speaker_mapping:
            return apply_speaker_mapping(segments, speaker_mapping)
        return segments

    # Flatten all words
    all_words: List[WordTimestamp] = []
    for seg in segments:
        all_words.extend(seg.words)

    if not all_words:
        logger.warning("No word timestamps found in segments. Retaining original segments.")
        return segments

    # Map words to speakers
    attributed_words = []
    for word in all_words:
        speaker = find_speaker_for_word(word, intervals)
        attributed_words.append((word, speaker))

    # Regroup contiguous words of the same speaker
    new_segments: List[TranscriptSegment] = []
    current_words: List[WordTimestamp] = []
    current_speaker: Optional[str] = None
    seg_id = 0

    for word, speaker in attributed_words:
        if current_speaker is None:
            current_speaker = speaker
            current_words = [word]
        elif speaker == current_speaker and (word.start - current_words[-1].end) <= max_pause_split_sec:
            current_words.append(word)
        else:
            # Emit current segment
            seg_text = " ".join(w.word for w in current_words).strip()
            new_segments.append(
                TranscriptSegment(
                    segment_id=seg_id,
                    speaker=current_speaker,
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=seg_text,
                    raw_text=seg_text,
                    words=current_words,
                )
            )
            seg_id += 1
            current_speaker = speaker
            current_words = [word]

    # Emit final segment
    if current_words and current_speaker is not None:
        seg_text = " ".join(w.word for w in current_words).strip()
        new_segments.append(
            TranscriptSegment(
                segment_id=seg_id,
                speaker=current_speaker,
                start=current_words[0].start,
                end=current_words[-1].end,
                text=seg_text,
                raw_text=seg_text,
                words=current_words,
            )
        )

    if speaker_mapping:
        new_segments = apply_speaker_mapping(new_segments, speaker_mapping)

    logger.info(f"Re-aligned into {len(new_segments)} speaker-attributed segments.")
    return new_segments


def apply_speaker_mapping(
    segments: List[TranscriptSegment],
    speaker_mapping: Dict[str, str],
) -> List[TranscriptSegment]:
    """Apply human-readable speaker names mapping (e.g. SPEAKER_00 -> 'Alice')."""
    for seg in segments:
        if seg.speaker in speaker_mapping:
            seg.speaker = speaker_mapping[seg.speaker]
    return segments
