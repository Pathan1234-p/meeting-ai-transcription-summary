"""
Decision extraction module.
Identifies consensus, approvals, and formal agreements from meeting transcripts.
"""

import re
import logging
from typing import List
from src.schemas import TranscriptSegment, DecisionItem
from src.cleanup.cleaner import TextCleaner

logger = logging.getLogger(__name__)

DECISION_CUES = [
    r"\bwe(?:\s+have|\s+'ve)?\s+decided\s+(?:to|that|on)\b",
    r"\bdecided\s+(?:to|that|on)\b",
    r"\bwe\s+agreed\s+(?:to|that|on)\b",
    r"\bagreed\s+(?:to|that|on)\b",
    r"\blet['’]s\s+go\s+with\b",
    r"\bwe(?:'ll|\s+will)\s+go\s+with\b",
    r"\bthe\s+plan\s+is\s+(?:to|that)\b",
    r"\bthe\s+decision\s+is\s+(?:to|that)\b",
    r"\bwe\s+settled\s+on\b",
    r"\bconsensus\s+is\s+(?:to|that)\b",
    r"\bwe\s+are\s+going\s+to\s+proceed\s+with\b",
    r"\bapproved\s+(?:the|to|that)\b",
]

DECISION_REGEX = re.compile("|".join(DECISION_CUES), re.IGNORECASE)

# Negative filter to ignore question queries like "did we decide?", "what did we decide?"
QUESTION_FILTER = re.compile(r"^(?:did|have|should|can|could|what|why|how)\s+", re.IGNORECASE)


class DecisionExtractor:
    """Extracts formal decisions and agreed-upon directions from meeting transcripts."""

    def __init__(self):
        self.cleaner = TextCleaner()

    def extract(self, segments: List[TranscriptSegment]) -> List[DecisionItem]:
        """
        Extract decisions from speaker-attributed transcript segments.

        Args:
            segments: List of TranscriptSegment objects.

        Returns:
            List of DecisionItem instances.
        """
        decisions: List[DecisionItem] = []
        seen_statements = set()
        decision_id = 0

        for seg in segments:
            text = seg.text or ""
            sentences = self.cleaner.segment_sentences(text)

            for sentence in sentences:
                clean_sent = sentence.strip()
                if not clean_sent or clean_sent.endswith("?"):
                    continue

                if QUESTION_FILTER.search(clean_sent):
                    continue

                match = DECISION_REGEX.search(clean_sent)
                if match:
                    # Clean statement
                    statement = clean_sent
                    # Avoid duplicate records
                    normalized_key = statement.lower().replace(" ", "")
                    if normalized_key in seen_statements:
                        continue
                    seen_statements.add(normalized_key)

                    confidence = 0.85
                    if "decided" in clean_sent.lower() or "consensus" in clean_sent.lower():
                        confidence = 0.92

                    decisions.append(
                        DecisionItem(
                            decision_id=decision_id,
                            statement=statement,
                            proposed_by=seg.speaker if seg.speaker != "Speaker unknown" else None,
                            timestamp=seg.start,
                            supporting_quote=clean_sent,
                            confidence=confidence,
                        )
                    )
                    decision_id += 1

        logger.info(f"Extracted {len(decisions)} decisions from {len(segments)} segments.")
        return decisions
