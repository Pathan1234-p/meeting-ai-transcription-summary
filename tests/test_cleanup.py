"""
Unit tests for text cleanup rules and spaCy segmentation.
"""

import pytest
from src.cleanup.cleaner import TextCleaner
from src.schemas import TranscriptSegment


@pytest.fixture(scope="module")
def cleaner():
    return TextCleaner()


def test_filler_word_removal(cleaner):
    """Test removing vocal pauses and filler words."""
    raw = "Um, so uh we need to, you know, sort of deploy the model."
    cleaned = cleaner.clean_text(raw)
    assert "um" not in cleaned.lower()
    assert "uh" not in cleaned.lower()
    assert "you know" not in cleaned.lower()
    assert "sort of" not in cleaned.lower()
    assert "deploy the model" in cleaned


def test_repeated_words_collapse(cleaner):
    """Test collapsing stutters and immediate word repetitions."""
    raw = "I I think we we we should consider that that option."
    cleaned = cleaner.clean_text(raw)
    assert "I I" not in cleaned
    assert "we we" not in cleaned
    assert "that that" not in cleaned
    assert "I think we should consider that option." in cleaned


def test_punctuation_and_capitalization(cleaner):
    """Test capitalization restoration and punctuation spacing."""
    raw = "hello world , this is a test . next sentence here ."
    cleaned = cleaner.clean_text(raw)
    assert cleaned.startswith("H")
    assert " ," not in cleaned
    assert " ." not in cleaned
    assert "test. Next" in cleaned


def test_spacy_sentence_segmentation(cleaner):
    """Test splitting multi-sentence transcripts cleanly."""
    text = "The sprint ends this Friday. Priya will lead the QA review. Everyone should attend."
    sentences = cleaner.segment_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "The sprint ends this Friday."
    assert sentences[1] == "Priya will lead the QA review."
    assert sentences[2] == "Everyone should attend."


def test_build_cleaned_transcript(cleaner):
    """Test assembling CleanedTranscript object."""
    segments = [
        TranscriptSegment(
            segment_id=0,
            speaker="Alice",
            start=0.0,
            end=4.5,
            text="Um, we we decided on the architecture.",
        ),
        TranscriptSegment(
            segment_id=1,
            speaker="Bob",
            start=4.6,
            end=8.0,
            text="Uh, yeah, Bob will deploy the database by Monday.",
        ),
    ]

    transcript = cleaner.build_cleaned_transcript(segments, duration_seconds=8.0)
    assert transcript.word_count > 0
    assert transcript.duration_seconds == 8.0
    assert set(transcript.speakers) == {"Alice", "Bob"}
    assert "um" not in transcript.full_text.lower()
    assert "we we" not in transcript.full_text.lower()
