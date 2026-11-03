"""
Text cleanup module for meeting speech transcripts.
Applies rule-based filler removal, repetition collapse, punctuation normalization,
and spaCy linguistic sentence segmentation.
"""

import re
import logging
from typing import List
import spacy
from src.schemas import TranscriptSegment, CleanedTranscript

logger = logging.getLogger(__name__)

# Primary conversational filler words and phrases
FILLER_WORDS_REGEX = re.compile(
    r"\b(um|uh|erm|er|ah|uhm|umm)\b[,]?\s*",
    flags=re.IGNORECASE
)

FILLER_PHRASES_REGEX = re.compile(
    r"\b(you know|sort of|kind of|i mean)\b[,]?\s*",
    flags=re.IGNORECASE
)

# Filler "like" when surrounded by pauses/commas or at clause boundaries
FILLER_LIKE_REGEX = re.compile(
    r"(,\s*like\s*,|\blike,\s*|\b(?<=[,\s])like(?=[,\s]))",
    flags=re.IGNORECASE
)

# Stuttering and repeated words (e.g. "I I", "the the", "we we we")
REPEATED_WORDS_REGEX = re.compile(
    r"\b(\w+)(?:\s+\1\b)+",
    flags=re.IGNORECASE
)

# Punctuation artifacts
MULTIPLE_SPACES_REGEX = re.compile(r"\s+")
MULTIPLE_COMMAS_REGEX = re.compile(r",\s*,+")
ORPHANED_PUNCTUATION_REGEX = re.compile(r"^[,\s]+|[,\s]+$")


class TextCleaner:
    """Cleans conversational speech transcripts into readable, well-punctuated text."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model for text cleanup: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model, disable=["ner"])
        except Exception as e:
            logger.warning(f"Could not load '{spacy_model}', attempting spacy blank('en'): {e}")
            self.nlp = spacy.blank("en")
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer")

    def clean_text(self, text: str) -> str:
        """
        Clean an individual string transcript.

        Args:
            text: Raw transcript text.

        Returns:
            Normalized, clean text.
        """
        if not text:
            return ""

        cleaned = text.strip()

        # 1. Remove filler words (um, uh, etc.)
        cleaned = FILLER_WORDS_REGEX.sub(" ", cleaned)

        # 2. Remove filler phrases (you know, sort of, kind of)
        cleaned = FILLER_PHRASES_REGEX.sub(" ", cleaned)

        # 3. Collapse repeated words / stutters (e.g. "we we" -> "we")
        # Match case-insensitively, keeping the first word's capitalization
        def collapse_repeat(match):
            words = match.group(0).split()
            return words[0]

        cleaned = REPEATED_WORDS_REGEX.sub(collapse_repeat, cleaned)

        # 4. Clean up punctuation artifacts
        cleaned = MULTIPLE_COMMAS_REGEX.sub(", ", cleaned)
        cleaned = MULTIPLE_SPACES_REGEX.sub(" ", cleaned).strip()
        cleaned = ORPHANED_PUNCTUATION_REGEX.sub("", cleaned).strip()

        # 5. Fix spaces before punctuation (e.g. "hello ," -> "hello,")
        cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)

        # 6. Ensure proper capitalization of sentences
        if cleaned:
            # Capitalize first letter if not already
            cleaned = cleaned[0].upper() + cleaned[1:]

        # Capitalize after sentence boundary (.!?)
        def capitalize_sentence(match):
            punct = match.group(1)
            char = match.group(2).upper()
            return f"{punct} {char}"

        cleaned = re.sub(r"([.!?])\s+([a-z])", capitalize_sentence, cleaned)

        return cleaned

    def clean_segments(self, segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Clean all segments in-place, preserving raw_text."""
        for seg in segments:
            if seg.raw_text is None:
                seg.raw_text = seg.text
            seg.text = self.clean_text(seg.raw_text)
        return segments

    def segment_sentences(self, text: str) -> List[str]:
        """Segment a long text block into linguistic sentences using spaCy."""
        if not text.strip():
            return []
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def build_cleaned_transcript(
        self,
        segments: List[TranscriptSegment],
        duration_seconds: float,
    ) -> CleanedTranscript:
        """
        Produce a CleanedTranscript object containing cleaned segments,
        aggregated full text, raw text, and speaker list.
        """
        cleaned_segments = self.clean_segments(segments)

        raw_parts = [s.raw_text or s.text for s in cleaned_segments if (s.raw_text or s.text)]
        clean_parts = [s.text for s in cleaned_segments if s.text]

        raw_text = " ".join(raw_parts)
        full_text = " ".join(clean_parts)
        words = full_text.split()

        unique_speakers = sorted(list({s.speaker for s in cleaned_segments if s.speaker}))

        return CleanedTranscript(
            full_text=full_text,
            raw_text=raw_text,
            segments=cleaned_segments,
            word_count=len(words),
            duration_seconds=round(duration_seconds, 2),
            speakers=unique_speakers,
        )
