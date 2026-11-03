"""
Data schemas for AI-Powered Meeting Documentation system.
Uses Pydantic v2 models for rigorous data validation and JSON serialization.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    """Word-level timestamp details from ASR."""
    word: str = Field(..., description="Transcribed word token")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    probability: float = Field(default=1.0, description="Word confidence score [0.0, 1.0]")


class TranscriptSegment(BaseModel):
    """Speaker-attributed transcript segment."""
    segment_id: int = Field(..., description="Monotonically increasing segment identifier")
    speaker: str = Field(default="Speaker unknown", description="Attributed speaker label or name")
    start: float = Field(..., description="Start timestamp in seconds")
    end: float = Field(..., description="End timestamp in seconds")
    text: str = Field(..., description="Cleaned transcript text for this segment")
    raw_text: Optional[str] = Field(default=None, description="Uncleaned original ASR text")
    words: List[WordTimestamp] = Field(default_factory=list, description="Word-level timestamps")


class CleanedTranscript(BaseModel):
    """Full meeting transcript with metadata and speaker segments."""
    full_text: str = Field(..., description="Concatenated cleaned text of the full meeting")
    raw_text: str = Field(..., description="Raw concatenated text prior to cleaning")
    segments: List[TranscriptSegment] = Field(default_factory=list, description="Segment list")
    word_count: int = Field(..., description="Total word count in cleaned transcript")
    duration_seconds: float = Field(..., description="Total duration of audio in seconds")
    speakers: List[str] = Field(default_factory=list, description="Unique speakers present")


class SummarySentence(BaseModel):
    """Sentence in abstractive summary with hallucination guard verification."""
    sentence: str = Field(..., description="Summary sentence text")
    faithfulness_score: float = Field(..., description="NLI entailment confidence score [0.0, 1.0]")
    status: str = Field(..., description="'Supported', 'Ambiguous', or 'Unsupported'")
    supporting_premise: Optional[str] = Field(default=None, description="Best matching source transcript sentence")


class MeetingSummary(BaseModel):
    """Extractive, abstractive, and executive summary outputs."""
    extractive_summary: List[str] = Field(default_factory=list, description="Key extracted sentences")
    abstractive_summary: str = Field(..., description="Fluent abstractive summary paragraph(s)")
    abstractive_sentences: List[SummarySentence] = Field(
        default_factory=list, 
        description="Sentence-level hallucination guard analysis"
    )
    executive_bullets: List[str] = Field(default_factory=list, description="High-level bullet points")
    compression_ratio: float = Field(..., description="Summary length / transcript length")
    hallucination_rate: float = Field(
        default=0.0, 
        description="Percentage of summary sentences flagged as unsupported"
    )


class DecisionItem(BaseModel):
    """Extracted meeting decision with speaker and evidence attribution."""
    decision_id: int = Field(..., description="Unique decision item identifier")
    statement: str = Field(..., description="Formal description of the decision reached")
    proposed_by: Optional[str] = Field(default=None, description="Speaker who proposed or agreed")
    timestamp: float = Field(..., description="Timestamp in seconds where decision was reached")
    supporting_quote: str = Field(..., description="Exact quote from transcript demonstrating agreement")
    confidence: float = Field(default=0.8, description="Extraction confidence score [0.0, 1.0]")


class ActionItem(BaseModel):
    """
    Extracted action item adhering strictly to explicit attribution rules:
    - Owner is assigned ONLY when explicitly stated in transcript (e.g. 'Priya, please...', 'I will...')
    - If ownership is ambiguous or unstated, owner MUST be set to 'Unassigned'.
    - Never infer or guess owners.
    """
    action_id: int = Field(..., description="Unique action item identifier")
    description: str = Field(..., description="Actionable task description")
    owner: str = Field(
        default="Unassigned", 
        description="Responsible person name if explicitly stated; strictly 'Unassigned' otherwise"
    )
    deadline: Optional[str] = Field(default=None, description="Explicit deadline if mentioned, else None")
    source_speaker: str = Field(..., description="Speaker who assigned or committed to the task")
    timestamp: float = Field(..., description="Timestamp in seconds of the action item mention")
    supporting_quote: str = Field(..., description="Exact supporting transcript quote")
    confidence: float = Field(default=0.8, description="Extraction confidence score [0.0, 1.0]")


class MeetingAnalysisResult(BaseModel):
    """Complete consolidated analysis output for a meeting."""
    meeting_id: str = Field(..., description="Identifier or filename for the meeting")
    audio_path: Optional[str] = Field(default=None, description="Source audio path")
    transcript: CleanedTranscript = Field(..., description="Cleaned and speaker-aligned transcript")
    summary: MeetingSummary = Field(..., description="Summaries with hallucination guard")
    decisions: List[DecisionItem] = Field(default_factory=list, description="List of decisions")
    action_items: List[ActionItem] = Field(default_factory=list, description="List of action items")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Pipeline configuration and timing metadata")
