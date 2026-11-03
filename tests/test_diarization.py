"""
Unit tests for speaker diarization fallback and word-speaker timestamp alignment.
"""

import pytest
from src.schemas import TranscriptSegment, WordTimestamp
from src.diarization.diarizer import SpeakerDiarizer, DiarizationInterval
from src.diarization.merger import (
    find_speaker_for_word,
    align_words_with_speakers,
    apply_speaker_mapping,
)


def test_diarizer_graceful_fallback():
    """Test that SpeakerDiarizer gracefully falls back when no token is present."""
    diarizer = SpeakerDiarizer(hf_token="")
    intervals = diarizer.diarize("dummy.wav")
    assert intervals == []


def test_find_speaker_for_word():
    """Test word-speaker matching by midpoint and overlap."""
    intervals = [
        DiarizationInterval(speaker="SPEAKER_00", start=0.0, end=3.0),
        DiarizationInterval(speaker="SPEAKER_01", start=3.2, end=6.0),
    ]

    # Word strictly inside SPEAKER_00
    w1 = WordTimestamp(word="Hello", start=0.5, end=1.0)
    assert find_speaker_for_word(w1, intervals) == "SPEAKER_00"

    # Word strictly inside SPEAKER_01
    w2 = WordTimestamp(word="World", start=3.5, end=4.0)
    assert find_speaker_for_word(w2, intervals) == "SPEAKER_01"

    # Word spanning boundary - midpoint test
    w3 = WordTimestamp(word="Boundary", start=2.8, end=3.1)  # mid=2.95 -> in SPEAKER_00
    assert find_speaker_for_word(w3, intervals) == "SPEAKER_00"


def test_align_words_with_speakers_and_mapping():
    """Test re-segmenting words and applying user speaker mapping."""
    words = [
        WordTimestamp(word="Good", start=0.0, end=0.4),
        WordTimestamp(word="morning", start=0.5, end=0.9),
        WordTimestamp(word="everyone.", start=1.0, end=1.4),
        WordTimestamp(word="Thanks,", start=2.0, end=2.4),
        WordTimestamp(word="let's", start=2.5, end=2.8),
        WordTimestamp(word="begin.", start=2.9, end=3.3),
    ]

    intervals = [
        DiarizationInterval(speaker="SPEAKER_00", start=0.0, end=1.6),
        DiarizationInterval(speaker="SPEAKER_01", start=1.8, end=4.0),
    ]

    raw_segments = [
        TranscriptSegment(
            segment_id=0,
            speaker="Speaker unknown",
            start=0.0,
            end=3.3,
            text="Good morning everyone. Thanks, let's begin.",
            words=words,
        )
    ]

    mapping = {
        "SPEAKER_00": "Anas",
        "SPEAKER_01": "Priya",
    }

    aligned = align_words_with_speakers(raw_segments, intervals, speaker_mapping=mapping)

    assert len(aligned) == 2
    assert aligned[0].speaker == "Anas"
    assert aligned[0].text == "Good morning everyone."
    assert aligned[1].speaker == "Priya"
    assert aligned[1].text == "Thanks, let's begin."
