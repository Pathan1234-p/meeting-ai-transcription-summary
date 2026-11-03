"""
Live integration test for WhisperTranscriber and TextCleaner on synthetic speech audio.
"""

import os
from src.audio.preprocessing import preprocess_audio
from src.asr.transcriber import WhisperTranscriber
from src.cleanup.cleaner import TextCleaner


def test_transcribe_and_clean():
    audio_path = "tests/demo_speech.wav"
    assert os.path.exists(audio_path), "demo_speech.wav must exist"

    # 1. Preprocess
    clean_wav, duration = preprocess_audio(audio_path)
    assert os.path.exists(clean_wav)
    assert duration > 1.0

    # 2. Transcribe using faster-whisper tiny for fast test
    transcriber = WhisperTranscriber(model_size="tiny", device="cpu", compute_type="int8")
    segments, elapsed = transcriber.transcribe(clean_wav)

    assert len(segments) > 0
    raw_text = " ".join(s.text for s in segments)
    print(f"\nRaw transcribed text ({elapsed:.2f}s):\n{raw_text}")

    # Verify word timestamps
    assert any(len(s.words) > 0 for s in segments)
    for seg in segments:
        for w in seg.words:
            assert w.start <= w.end

    # 3. Text Cleaner
    cleaner = TextCleaner()
    transcript = cleaner.build_cleaned_transcript(segments, duration_seconds=duration)

    print(f"\nCleaned transcript ({transcript.word_count} words):\n{transcript.full_text}")
    assert len(transcript.full_text) > 0
    assert transcript.duration_seconds > 0


if __name__ == "__main__":
    test_transcribe_and_clean()
