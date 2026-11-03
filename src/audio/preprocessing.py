"""
Audio preprocessing module.
Standardizes any audio/video input into 16 kHz mono 16-bit PCM WAV.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import imageio_ffmpeg
from pydub import AudioSegment
import soundfile as sf

# Configure pydub to use the bundled imageio-ffmpeg binary
ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()
AudioSegment.converter = ffmpeg_binary

logger = logging.getLogger(__name__)


def preprocess_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
) -> Tuple[str, float]:
    """
    Standardize an input audio or video file to 16 kHz mono WAV format.

    Args:
        input_path: Path to the input audio/video file.
        output_path: Destination path for the converted WAV. If None, creates next to input.
        target_sample_rate: Desired sampling rate (default 16000 Hz for Whisper).
        target_channels: Desired channels (1 = mono).

    Returns:
        Tuple of (normalized_wav_path, duration_seconds)
    """
    input_file = Path(input_path).resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {input_path}")

    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_16k_mono.wav")
    else:
        output_file = Path(output_path).resolve()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preprocessing audio: {input_file} -> {output_file}")

    # Check if file is already 16kHz mono WAV to avoid re-encoding
    if input_file.suffix.lower() == ".wav":
        try:
            info = sf.info(str(input_file))
            if info.samplerate == target_sample_rate and info.channels == target_channels:
                logger.info("Audio already in desired 16 kHz mono WAV format.")
                # If target is different file path, copy or write
                if output_file != input_file:
                    data, sr = sf.read(str(input_file))
                    sf.write(str(output_file), data, sr, subtype="PCM_16")
                return str(output_file), info.duration
        except Exception as e:
            logger.debug(f"Direct soundfile check failed, falling back to pydub: {e}")

    # Load audio via pydub with ffmpeg support
    try:
        audio = AudioSegment.from_file(str(input_file))
    except Exception as e:
        logger.error(f"Failed to load audio with pydub: {e}")
        raise RuntimeError(f"Unable to decode audio from {input_path}: {e}")

    # Resample and convert to mono
    audio = audio.set_frame_rate(target_sample_rate)
    audio = audio.set_channels(target_channels)
    audio = audio.set_sample_width(2)  # 16-bit

    # Export to WAV
    audio.export(str(output_file), format="wav")
    duration_seconds = len(audio) / 1000.0

    logger.info(f"Successfully processed audio: duration={duration_seconds:.2f}s, rate={target_sample_rate}Hz, channels={target_channels}")
    return str(output_file), duration_seconds
