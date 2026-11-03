"""
ASR Evaluation module.
Evaluates Word Error Rate (WER) and Character Error Rate (CER) across meetings,
and benchmarks Whisper model sizes (tiny, base, small) on accuracy vs. Real-Time Factor (RTF).
"""

import time
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import jiwer
from src.asr.transcriber import WhisperTranscriber
from src.audio.preprocessing import preprocess_audio
from src.cleanup.cleaner import TextCleaner

logger = logging.getLogger("ASREval")


def is_model_cached(model_size: str) -> bool:
    """
    Check whether a faster-whisper model's weights are fully present in the
    local HuggingFace cache so we don't hang waiting for a network download.
    """
    hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_dir = hf_cache / f"models--Systran--faster-whisper-{model_size}"
    if not model_dir.exists():
        return False
    # Look for model.bin in blobs/ or any snapshot subdir
    for candidate in model_dir.rglob("model.bin"):
        try:
            if candidate.stat().st_size > 10_000_000:  # >10MB → real weights
                return True
        except OSError:
            pass
    return False

# Standard normalization for fair ASR evaluation
asr_transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])


def normalize_text_for_wer(text: str) -> str:
    """Standardize text for objective WER/CER computation."""
    # Strip speaker prefixes like "David:" or "SPEAKER_00:"
    lines = []
    for line in text.splitlines():
        if ":" in line:
            parts = line.split(":", 1)
            lines.append(parts[1].strip())
        else:
            lines.append(line.strip())
    combined = " ".join(lines)
    return asr_transform(combined)


def evaluate_asr_meeting(
    audio_path: str,
    reference_transcript_path: str,
    transcriber: WhisperTranscriber,
) -> Dict[str, Any]:
    """Evaluate ASR performance on a single meeting."""
    clean_wav, duration_sec = preprocess_audio(audio_path)

    t0 = time.time()
    segments, _ = transcriber.transcribe(clean_wav)
    transcription_time = time.time() - t0

    hypothesis_raw = " ".join(s.text for s in segments)
    with open(reference_transcript_path, "r", encoding="utf-8") as f:
        reference_raw = f.read()

    ref_norm = normalize_text_for_wer(reference_raw)
    hyp_norm = normalize_text_for_wer(hypothesis_raw)

    wer = jiwer.wer(ref_norm, hyp_norm)
    cer = jiwer.cer(ref_norm, hyp_norm)
    rtf = round(transcription_time / max(duration_sec, 0.001), 3)

    return {
        "duration_sec": round(duration_sec, 2),
        "transcription_time_sec": round(transcription_time, 2),
        "rtf": rtf,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "reference_norm": ref_norm,
        "hypothesis_norm": hyp_norm,
        "hypothesis_raw": hypothesis_raw,
    }


def benchmark_whisper_sizes(
    meetings_dirs: List[Path],
    model_sizes: List[str] = ["tiny", "base", "small"],
) -> pd.DataFrame:
    """Benchmark different Whisper model sizes on WER, CER, and speed.
    Only evaluates model sizes that are fully present in the local HF cache
    to avoid indefinite hangs on network downloads.
    """
    records = []

    for size in model_sizes:
        if not is_model_cached(size):
            logger.warning(
                f"Skipping Whisper '{size}' — model.bin not found in local cache. "
                f"Download it manually or re-run after it completes."
            )
            continue

        logger.info(f"Benchmarking Whisper model size: {size}...")
        transcriber = WhisperTranscriber(model_size=size)

        total_audio_sec = 0.0
        total_time_sec = 0.0
        wers = []
        cers = []

        for m_dir in meetings_dirs:
            audio_path = str(m_dir / "audio.wav")
            ref_path = str(m_dir / "reference_transcript.txt")
            if not Path(audio_path).exists():
                continue

            metrics = evaluate_asr_meeting(audio_path, ref_path, transcriber)
            total_audio_sec += metrics["duration_sec"]
            total_time_sec += metrics["transcription_time_sec"]
            wers.append(metrics["wer"])
            cers.append(metrics["cer"])

        avg_wer = sum(wers) / len(wers) if wers else 0.0
        avg_cer = sum(cers) / len(cers) if cers else 0.0
        overall_rtf = round(total_time_sec / max(total_audio_sec, 0.001), 3)

        records.append({
            "Model Size": size,
            "Mean WER (%)": round(avg_wer * 100, 2),
            "Mean CER (%)": round(avg_cer * 100, 2),
            "Total Audio (s)": round(total_audio_sec, 1),
            "Total Time (s)": round(total_time_sec, 1),
            "Real-Time Factor (RTF)": overall_rtf,
            "Speedup vs Real-time": f"{round(1.0 / max(overall_rtf, 0.001), 1)}x",
        })

    if not records:
        logger.error("No Whisper models were found in local cache — benchmark table is empty!")
    return pd.DataFrame(records)

