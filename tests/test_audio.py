"""
Unit tests for audio preprocessing and chunking.
"""

import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
import pytest
from src.audio.preprocessing import preprocess_audio
from src.audio.chunking import chunk_audio


@pytest.fixture
def sample_wav_file():
    """Create a temporary 44100Hz stereo WAV file."""
    sr = 44100
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate 440 Hz sine wave for two channels
    left = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (0.5 * np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    stereo = np.column_stack((left, right))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    wavfile.write(temp_path, sr, stereo)
    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_preprocess_audio(sample_wav_file):
    """Test audio conversion to 16kHz mono WAV."""
    out_wav, duration = preprocess_audio(sample_wav_file)

    assert os.path.exists(out_wav)
    assert abs(duration - 2.0) < 0.1

    sr, data = wavfile.read(out_wav)
    assert sr == 16000
    assert len(data.shape) == 1  # Mono

    # Clean up output
    if os.path.exists(out_wav):
        os.remove(out_wav)


def test_chunk_audio_short(sample_wav_file):
    """Test chunking on short audio where no split is needed."""
    chunks = chunk_audio(sample_wav_file, chunk_length_sec=10.0, overlap_sec=1.0)
    assert len(chunks) == 1
    assert chunks[0]["start_sec"] == 0.0
    assert abs(chunks[0]["end_sec"] - 2.0) < 0.1


def test_chunk_audio_long():
    """Test chunking on a synthesized 6-second file with 3-second chunks and 1-second overlap."""
    sr = 16000
    duration = 6.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    wavfile.write(temp_path, sr, data)

    try:
        chunks = chunk_audio(temp_path, chunk_length_sec=3.0, overlap_sec=1.0)
        # Expected:
        # chunk 0: 0.0 to 3.0
        # chunk 1: 2.0 to 5.0
        # chunk 2: 4.0 to 6.0
        assert len(chunks) >= 2
        for chunk in chunks:
            assert os.path.exists(chunk["chunk_path"])
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
