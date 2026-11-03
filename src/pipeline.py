"""
End-to-end Meeting AI Pipeline orchestrator and CLI runner.
Integrates Audio Preprocessing, ASR, Diarization, Cleanup, Summarization,
Hallucination Guard, and Decision/Action-Item Extraction into a unified workflow.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.schemas import (
    MeetingAnalysisResult,
    MeetingSummary,
    CleanedTranscript,
    TranscriptSegment,
    DecisionItem,
    ActionItem,
)
from src.config import config
from src.audio.preprocessing import preprocess_audio
from src.audio.chunking import chunk_audio
from src.asr.transcriber import WhisperTranscriber
from src.diarization.diarizer import SpeakerDiarizer
from src.diarization.merger import align_words_with_speakers, apply_speaker_mapping
from src.cleanup.cleaner import TextCleaner
from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.abstractive import AbstractiveSummarizer
from src.summarization.hallucination_guard import HallucinationGuard
from src.extraction.decisions import DecisionExtractor
from src.extraction.action_items import ActionItemExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("MeetingPipeline")


class MeetingPipeline:
    """End-to-end pipeline orchestrating speech-to-documentation workflow."""

    def __init__(
        self,
        whisper_model_size: Optional[str] = None,
        whisper_device: Optional[str] = None,
        hf_token: Optional[str] = None,
        enable_abstractive: bool = True,
        enable_hallucination_guard: bool = True,
    ):
        self.whisper_model_size = whisper_model_size or config.whisper_model_size
        self.whisper_device = whisper_device or config.whisper_device
        self.hf_token = hf_token or config.hf_token
        self.enable_abstractive = enable_abstractive
        self.enable_hallucination_guard = enable_hallucination_guard

        logger.info(f"Initializing pipeline components (ASR={self.whisper_model_size})...")
        self.transcriber = WhisperTranscriber(
            model_size=self.whisper_model_size,
            device=self.whisper_device,
        )
        self.diarizer = SpeakerDiarizer(hf_token=self.hf_token)
        self.cleaner = TextCleaner()
        self.extractive_summarizer = ExtractiveSummarizer()
        self.abstractive_summarizer = AbstractiveSummarizer() if enable_abstractive else None
        self.hallucination_guard = HallucinationGuard() if enable_hallucination_guard else None
        self.decision_extractor = DecisionExtractor()
        self.action_extractor = ActionItemExtractor()

    def process(
        self,
        audio_path: str,
        speaker_mapping: Optional[Dict[str, str]] = None,
        meeting_id: Optional[str] = None,
    ) -> MeetingAnalysisResult:
        """
        Run the full end-to-end meeting analysis pipeline on an audio file.

        Args:
            audio_path: Path to the input audio/video file.
            speaker_mapping: Optional dict mapping speaker IDs (e.g. SPEAKER_00) to names.
            meeting_id: Optional unique meeting identifier.

        Returns:
            MeetingAnalysisResult instance.
        """
        start_pipeline = time.time()
        file_path = Path(audio_path).resolve()
        meeting_name = meeting_id or file_path.stem

        logger.info(f"=== Starting Meeting AI Pipeline for: {meeting_name} ===")

        # Step 1: Audio Preprocessing to 16kHz mono WAV
        t0 = time.time()
        norm_wav, duration_sec = preprocess_audio(str(file_path))
        t_prep = time.time() - t0

        # Step 2: Audio Chunking (if audio exceeds 15 minutes)
        chunks = chunk_audio(norm_wav, chunk_length_sec=900.0, overlap_sec=2.0)

        # Step 3: ASR Transcription
        t0 = time.time()
        all_raw_segments: List[TranscriptSegment] = []
        for chunk in chunks:
            seg_list, _ = self.transcriber.transcribe(
                chunk["chunk_path"],
                time_offset=chunk["offset_sec"],
            )
            all_raw_segments.extend(seg_list)
        t_asr = time.time() - t0

        # Step 4: Speaker Diarization
        t0 = time.time()
        diar_intervals = self.diarizer.diarize(norm_wav)
        t_diar = time.time() - t0

        # Step 5: Align Words with Speakers
        aligned_segments = align_words_with_speakers(
            all_raw_segments,
            diar_intervals,
            speaker_mapping=speaker_mapping,
        )

        # Step 6: Text Cleanup & Aggregation
        t0 = time.time()
        transcript = self.cleaner.build_cleaned_transcript(aligned_segments, duration_sec)
        t_clean = time.time() - t0

        # Step 7: Extractive Summarization
        t0 = time.time()
        extractive_summary = self.extractive_summarizer.summarize(
            transcript.full_text,
            ratio=config.extractive_ratio,
        )
        t_ext = time.time() - t0

        # Step 8: Abstractive Summarization & Executive Bullets
        t0 = time.time()
        if self.abstractive_summarizer:
            abstractive_text, executive_bullets = self.abstractive_summarizer.summarize(
                transcript.full_text
            )
        else:
            abstractive_text = " ".join(extractive_summary)
            executive_bullets = [f"Summary: {s}" for s in extractive_summary[:3]]
        t_abs = time.time() - t0

        # Step 9: Hallucination Guard
        t0 = time.time()
        if self.hallucination_guard:
            verified_sentences, hallucination_rate = self.hallucination_guard.verify_summary(
                abstractive_text,
                transcript.full_text,
            )
        else:
            verified_sentences = []
            hallucination_rate = 0.0
        t_nli = time.time() - t0

        # Compute compression ratio
        transcript_words = transcript.word_count or 1
        summary_words = len(abstractive_text.split())
        compression_ratio = round(summary_words / transcript_words, 3)

        meeting_summary = MeetingSummary(
            extractive_summary=extractive_summary,
            abstractive_summary=abstractive_text,
            abstractive_sentences=verified_sentences,
            executive_bullets=executive_bullets,
            compression_ratio=compression_ratio,
            hallucination_rate=hallucination_rate,
        )

        # Step 10: Decision Extraction
        t0 = time.time()
        decisions = self.decision_extractor.extract(transcript.segments)
        t_dec = time.time() - t0

        # Step 11: Action Item Extraction (Strict explicit attribution)
        t0 = time.time()
        action_items = self.action_extractor.extract(transcript.segments)
        t_act = time.time() - t0

        total_elapsed = time.time() - start_pipeline
        rtf = round(total_elapsed / max(duration_sec, 0.001), 3)

        metadata = {
            "duration_seconds": round(duration_sec, 2),
            "total_processing_seconds": round(total_elapsed, 2),
            "real_time_factor": rtf,
            "whisper_model_size": self.whisper_model_size,
            "device": self.whisper_device,
            "timings": {
                "audio_preprocessing_sec": round(t_prep, 2),
                "asr_transcription_sec": round(t_asr, 2),
                "diarization_sec": round(t_diar, 2),
                "cleanup_sec": round(t_clean, 2),
                "extractive_summarization_sec": round(t_ext, 2),
                "abstractive_summarization_sec": round(t_abs, 2),
                "hallucination_guard_sec": round(t_nli, 2),
                "decision_extraction_sec": round(t_dec, 2),
                "action_extraction_sec": round(t_act, 2),
            },
        }

        logger.info(
            f"=== Pipeline Completed: {len(transcript.segments)} segments, "
            f"{len(decisions)} decisions, {len(action_items)} actions in {total_elapsed:.2f}s (RTF={rtf}) ==="
        )

        return MeetingAnalysisResult(
            meeting_id=meeting_name,
            audio_path=str(file_path),
            transcript=transcript,
            summary=meeting_summary,
            decisions=decisions,
            action_items=action_items,
            metadata=metadata,
        )

    @staticmethod
    def export_json(result: MeetingAnalysisResult, output_path: str):
        """Export analysis result to structured JSON file."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.model_dump_json(indent=2))
        logger.info(f"Exported JSON result to: {out_file}")

    @staticmethod
    def export_markdown(result: MeetingAnalysisResult, output_path: str):
        """Export analysis result to a clean, formatted Markdown report."""
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        md = []
        md.append(f"# Meeting Documentation: {result.meeting_id}\n")
        md.append(f"**Duration**: {result.transcript.duration_seconds:.1f}s | "
                  f"**Words**: {result.transcript.word_count} | "
                  f"**Processing Time**: {result.metadata.get('total_processing_seconds', 0)}s | "
                  f"**RTF**: {result.metadata.get('real_time_factor', 0)}\n")

        md.append("## Executive Summary\n")
        for bullet in result.summary.executive_bullets:
            md.append(f"- {bullet}")
        md.append("\n")

        md.append("## Abstractive Summary\n")
        md.append(f"{result.summary.abstractive_summary}\n")
        if result.summary.abstractive_sentences:
            md.append(f"> **Hallucination Rate**: {result.summary.hallucination_rate * 100:.1f}%\n")

        md.append("## Extractive Key Points\n")
        for s in result.summary.extractive_summary:
            md.append(f"- {s}")
        md.append("\n")

        md.append("## Decisions Log\n")
        if result.decisions:
            md.append("| # | Decision Statement | Proposed By | Timestamp | Supporting Quote |")
            md.append("|---|---|---|---|---|")
            for d in result.decisions:
                proposer = d.proposed_by or "Consensus"
                ts = f"{d.timestamp:.1f}s"
                quote = d.supporting_quote.replace("|", "/")
                md.append(f"| {d.decision_id} | {d.statement} | {proposer} | {ts} | *\"{quote}\"* |")
        else:
            md.append("*No formal decisions recorded.*\n")
        md.append("\n")

        md.append("## Action Items (Strict Explicit Ownership)\n")
        if result.action_items:
            md.append("| # | Task Description | Owner | Deadline | Speaker | Supporting Quote |")
            md.append("|---|---|---|---|---|---|")
            for a in result.action_items:
                owner = f"**{a.owner}**" if a.owner != "Unassigned" else "*Unassigned*"
                dl = a.deadline or "-"
                quote = a.supporting_quote.replace("|", "/")
                md.append(f"| {a.action_id} | {a.description} | {owner} | {dl} | {a.source_speaker} | *\"{quote}\"* |")
        else:
            md.append("*No action items detected.*\n")
        md.append("\n")

        md.append("## Speaker-Attributed Transcript\n")
        for seg in result.transcript.segments:
            md.append(f"**[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}**: {seg.text}\n")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
        logger.info(f"Exported Markdown report to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Meeting Documentation Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input audio/video file")
    parser.add_argument("--out", "-o", default="results", help="Directory for output reports")
    parser.add_argument("--model-size", default="small", choices=["tiny", "base", "small", "medium"], help="Whisper ASR model size")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for speaker diarization")
    parser.add_argument("--speaker-map", default=None, help="JSON string for speaker mapping e.g. '{\"SPEAKER_00\": \"Anas\"}'")

    args = parser.parse_args()

    speaker_map = None
    if args.speaker_map:
        try:
            speaker_map = json.loads(args.speaker_map)
        except Exception as e:
            logger.warning(f"Could not parse speaker-map JSON: {e}")

    pipeline = MeetingPipeline(
        whisper_model_size=args.model_size,
        hf_token=args.hf_token,
    )

    result = pipeline.process(
        audio_path=args.input,
        speaker_mapping=speaker_map,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{result.meeting_id}_results.json"
    md_path = out_dir / f"{result.meeting_id}_report.md"

    MeetingPipeline.export_json(result, str(json_path))
    MeetingPipeline.export_markdown(result, str(md_path))

    print(f"\n[DONE] Pipeline successfully executed!")
    print(f"Structured JSON: {json_path}")
    print(f"Markdown Report: {md_path}")


if __name__ == "__main__":
    main()
