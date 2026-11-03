"""
Action item extraction module adhering strictly to explicit owner attribution rules.
Never infers or guesses owners:
- Assigns owner ONLY when explicitly stated in the transcript (e.g. 'Priya, please...', 'I will...')
- Ambiguous or collective tasks are strictly assigned to 'Unassigned'.
"""

import re
import logging
from typing import List, Optional, Tuple
import spacy
from src.schemas import TranscriptSegment, ActionItem
from src.cleanup.cleaner import TextCleaner

logger = logging.getLogger(__name__)

# Non-person proper nouns to avoid false owner extraction
NON_PERSON_ENTITIES = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "today", "tomorrow", "yesterday",
    "postgresql", "postgres", "mysql", "redis", "docker", "kubernetes", "aws",
    "gcp", "azure", "linux", "windows", "python", "javascript", "react", "fastapi",
    "jira", "github", "gitlab", "slack", "sprint", "q1", "q2", "q3", "q4", "team",
}

# 1. Direct Assignment to Named Person (e.g. "Priya, please update the docs", "Bob will deploy")
DIRECT_ASSIGNMENT_PATTERNS = [
    re.compile(
        r"\b(?P<owner>[A-Z][a-z]+)[,\s]+(?:can you|could you|please|will you|you should|handle|take ownership of|take care of)\s+(?P<task>[^.?!]+)",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(?P<owner>[A-Z][a-z]+)\s+(?:will|is assigned to|is responsible for|takes on|is going to|to handle)\s+(?P<task>[^.?!]+)",
        re.IGNORECASE
    ),
    re.compile(
        r"\blet['’]s\s+have\s+(?P<owner>[A-Z][a-z]+)\s+(?P<task>[^.?!]+)",
        re.IGNORECASE
    ),
]

# 2. First-Person Commitment (e.g. "I will prepare the presentation", "I'll follow up")
FIRST_PERSON_PATTERNS = [
    re.compile(
        r"\b(?:I\s+will|I['’]ll|I['’]m\s+going\s+to|I\s+can\s+handle|I\s+will\s+take\s+care\s+of|I['’]ll\s+take\s+ownership\s+of|I\s+can\s+send)\s+(?P<task>[^.?!]+)",
        re.IGNORECASE
    ),
]

# 3. Collective / Passive / Ambiguous Actions (Strictly "Unassigned")
UNASSIGNED_ACTION_PATTERNS = [
    re.compile(
        r"\b(?:we\s+need\s+to|we\s+should|we\s+must|we\s+have\s+to|someone\s+needs\s+to|action\s+item\s+(?:is\s+)?to|let['’]s\s+make\s+sure\s+to|make\s+sure\s+to|don['’]t\s+forget\s+to)\s+(?P<task>[^.?!]+)",
        re.IGNORECASE
    ),
]

# Deadline extraction pattern
DEADLINE_PREP_REGEX = re.compile(
    r"\b(?:by|before|due(?:\s+date)?|deadline(?:\s+is)?|until|no later than)\s+((?:this|next)?\s*(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|tomorrow|today|EOD|end of day|end of the week|next week|end of month|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+|\d{1,2}/\d{1,2}))\b",
    re.IGNORECASE
)

DEADLINE_STANDALONE_REGEX = re.compile(
    r"\b((?:this|next)\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|tomorrow|today)\b",
    re.IGNORECASE
)


class ActionItemExtractor:
    """Extracts meeting action items with strict explicit owner attribution."""

    def __init__(self):
        self.cleaner = TextCleaner()

    def _extract_deadline(self, text: str) -> Tuple[Optional[str], str]:
        """Extract deadline phrase if present and return (deadline, text)."""
        m_prep = DEADLINE_PREP_REGEX.search(text)
        if m_prep:
            return m_prep.group(1).strip(), text

        m_stand = DEADLINE_STANDALONE_REGEX.search(text)
        if m_stand:
            return m_stand.group(1).strip(), text

        return None, text

    def _is_valid_person_name(self, name: str) -> bool:
        """Check whether candidate name is a valid person name, not a day, tech tool, or pronoun."""
        if not name:
            return False
        clean = name.strip().lower()
        if clean in NON_PERSON_ENTITIES:
            return False
        if clean in ["we", "you", "they", "someone", "everyone", "nobody", "it", "this"]:
            return False
        return len(clean) >= 2

    def extract(self, segments: List[TranscriptSegment]) -> List[ActionItem]:
        """
        Extract action items from transcript segments.

        Owner Attribution Rules:
        1. If a person is explicitly addressed or assigned ('Priya, please...', 'Alice will...'):
           -> owner = Person Name
        2. If speaker makes a first-person commitment ('I will...'):
           -> owner = Speaker Name (if speaker is identified and known), else 'Unassigned'
        3. If task is collective or unassigned ('We need to...', 'Someone should...'):
           -> owner = 'Unassigned'
        4. NEVER infer or guess owners.

        Returns:
            List of ActionItem instances.
        """
        action_items: List[ActionItem] = []
        seen_descriptions = set()
        action_id = 0

        for seg in segments:
            text = seg.text or ""
            sentences = self.cleaner.segment_sentences(text)

            for sentence in sentences:
                clean_sent = sentence.strip()
                if not clean_sent or clean_sent.endswith("?"):
                    continue

                matched_item = None

                # Rule 1: Check direct assignment to named person
                for pattern in DIRECT_ASSIGNMENT_PATTERNS:
                    m = pattern.search(clean_sent)
                    if m:
                        candidate_owner = m.group("owner").strip()
                        if self._is_valid_person_name(candidate_owner):
                            task = m.group("task").strip()
                            deadline, _ = self._extract_deadline(clean_sent)
                            matched_item = (task, candidate_owner.capitalize(), deadline, 0.90)
                            break

                # Rule 2: Check first-person commitment ("I will...")
                if matched_item is None:
                    for pattern in FIRST_PERSON_PATTERNS:
                        m = pattern.search(clean_sent)
                        if m:
                            task = m.group("task").strip()
                            deadline, _ = self._extract_deadline(clean_sent)
                            # If speaker is known and valid, attribute to speaker
                            if seg.speaker and seg.speaker != "Speaker unknown" and not seg.speaker.startswith("SPEAKER_"):
                                owner = seg.speaker
                                conf = 0.88
                            else:
                                # Speaker is unknown or raw ID; rule states owner must be explicitly named in transcript
                                owner = "Unassigned"
                                conf = 0.82
                            matched_item = (task, owner, deadline, conf)
                            break

                # Rule 3: Collective / Passive / Ambiguous directives
                if matched_item is None:
                    for pattern in UNASSIGNED_ACTION_PATTERNS:
                        m = pattern.search(clean_sent)
                        if m:
                            task = m.group("task").strip()
                            deadline, _ = self._extract_deadline(clean_sent)
                            # Strictly Unassigned: NEVER guess
                            matched_item = (task, "Unassigned", deadline, 0.85)
                            break

                if matched_item:
                    raw_task, owner, deadline, confidence = matched_item
                    # Clean task description
                    description = raw_task.strip()
                    if description.endswith("."):
                        description = description[:-1]
                    # Capitalize description
                    if description:
                        description = description[0].upper() + description[1:]

                    # Deduplication
                    norm_key = description.lower()
                    if norm_key in seen_descriptions:
                        continue
                    seen_descriptions.add(norm_key)

                    action_items.append(
                        ActionItem(
                            action_id=action_id,
                            description=description,
                            owner=owner,
                            deadline=deadline,
                            source_speaker=seg.speaker,
                            timestamp=seg.start,
                            supporting_quote=clean_sent,
                            confidence=confidence,
                        )
                    )
                    action_id += 1

        logger.info(f"Extracted {len(action_items)} action items ({sum(1 for a in action_items if a.owner != 'Unassigned')} assigned).")
        return action_items
