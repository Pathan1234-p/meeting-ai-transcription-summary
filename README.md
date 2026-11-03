# AI-Powered Meeting Transcription & Summary Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-green.svg)](https://github.com/SYSTRAN/faster-whisper)

A production-grade, CPU-optimized meeting documentation prototype and rigorous evaluation benchmark suite. Converts raw meeting audio/video recordings into clean, speaker-attributed transcripts, dual extractive and abstractive summaries with NLI-powered hallucination verification, formal decisions, and strictly-attributed action items.

---

## Key Capabilities

1. **Robust Audio Ingestion & Preprocessing**:
   - Automatic format normalization (MP3, WAV, M4A, FLAC, OGG, MP4) into 16 kHz mono 16-bit PCM WAV.
   - Long-file chunking with a 2-second overlap for memory-safe processing of hour-long sessions.
   - Bundled FFmpeg adapter via `imageio-ffmpeg` (zero external PATH configuration required on Windows).

2. **Accurate Speech-to-Text (`faster-whisper`)**:
   - CTranslate2 Whisper implementation with `int8` quantization for fast CPU execution (Real-Time Factor < 0.3x).
   - Word-level timestamps and Silero VAD filtering to remove silence and background noise.
   - Configurable model size: `tiny`, `base`, `small`, `medium`.

3. **Speaker Diarization & Word Alignment**:
   - `pyannote.audio` pipeline integration (`pyannote/speaker-diarization-3.1`).
   - Graceful fallback: If Hugging Face token or model is unavailable, logs a warning and marks speakers as `"Speaker unknown"` without crashing.
   - Temporal interval alignment mapping transcribed words to speaker turns.
   - User-defined speaker re-mapping dictionary (e.g., `SPEAKER_00` $\rightarrow$ `"Anas"`).

4. **Conversational Text Cleanup & Segmentation**:
   - Rule-based regex removing vocal fillers (`"um"`, `"uh"`, `"you know"`, `"like"`).
   - Collapse of stutters and immediate word repetitions (`"I I think we we"` $\rightarrow$ `"I think we"`).
   - Punctuation and casing restoration with linguistic sentence boundary detection via spaCy (`en_core_web_sm`).

5. **Dual Extractive & Abstractive Summarization**:
   - **Extractive**: Graph-based TextRank + Maximal Marginal Relevance (MMR) sentence clustering using `sentence-transformers` (`all-MiniLM-L6-v2`).
   - **Abstractive**: Hierarchical chunk-reduce summarization using Hugging Face transformer models (`sshleifer/distilbart-cnn-12-6` / `facebook/bart-large-cnn`) or pluggable API-based LLMs.
   - **Executive Bullets**: Automated generation of high-level bulleted briefs.

6. **NLI Hallucination Guard**:
   - Cross-verifies every abstractive summary sentence against source transcript premises using an NLI cross-encoder (`cross-encoder/nli-deberta-v3-small`).
   - Flags claims as **Supported** (Entailment), **Ambiguous** (Neutral), or **Unsupported** (Hallucination).
   - Automatically computes overall Hallucination Rate.

7. **Strict Action Item Attribution Standard**:
   - **Zero-Guessing Rule**: Assigns an owner **ONLY** when explicitly stated in the transcript (first-person commitment like `"I will do X"` or second-person direct assignment like `"Priya, please do X"`).
   - Ambiguous or collective tasks (`"We should update docs"`, `"Someone needs to test"`) are strictly tagged as `"Unassigned"`.
   - Extracts: Description, Owner, Deadline, Source Speaker, Timestamp, and Supporting Quote.

8. **Interactive Streamlit Web UI**:
   - Audio file upload and integrated playback player.
   - Real-time diagnostics (Duration, Word Count, RTF, Decisions, Actions).
   - Interactive speaker transcript timeline.
   - Side-by-side summary comparison with Hallucination Guard badges and premise citations.
   - Decisions Log and Action Items table.
   - One-click export to structured JSON and formatted Markdown reports.

---

## System Architecture

```
meeting-ai/
├── app/
│   └── app.py                     # Streamlit Interactive Web Application
├── src/
│   ├── audio/
│   │   ├── preprocessing.py       # Audio conversion to 16kHz mono WAV
│   │   └── chunking.py            # Long audio chunking with overlap
│   ├── asr/
│   │   └── transcriber.py         # faster-whisper wrapper with word timestamps
│   ├── diarization/
│   │   ├── diarizer.py            # pyannote wrapper with graceful fallback
│   │   └── merger.py              # Word-speaker timestamp alignment
│   ├── cleanup/
│   │   └── cleaner.py             # Filler removal, stutter collapse, spaCy segmentation
│   ├── summarization/
│   │   ├── extractive.py          # TextRank + MMR semantic clustering
│   │   ├── abstractive.py         # Hierarchical chunk-reduce abstractive summarizer
│   │   └── hallucination_guard.py # NLI sentence-level faithfulness verification
│   ├── extraction/
│   │   ├── decisions.py           # Formal decision & consensus extraction
│   │   └── action_items.py        # Strict explicit action item attribution
│   ├── config.py                  # Global settings and environment loader
│   ├── schemas.py                 # Pydantic v2 data models
│   └── pipeline.py                # End-to-end pipeline orchestrator & CLI
├── evaluation/
│   ├── datasets/                  # 5 Benchmark meetings (audio, transcript, references)
│   ├── asr_eval.py                # WER, CER, Whisper model size benchmark
│   ├── summary_eval.py            # ROUGE-1/2/L, BERTScore, compression ratio
│   ├── factuality_eval.py         # NLI faithfulness and hallucination rate
│   ├── completeness_eval.py       # Key points checklist & attribution accuracy
│   ├── error_analysis.py          # Error categorization & propagation analysis
│   └── run_all.py                 # Master single-command evaluation runner
├── reports/                       # Generated CSV tables, figures (.png), final report
├── tests/                         # Pytest unit & integration test suite
├── requirements.txt               # Dependencies
├── .env.example                   # Environment configuration template
└── README.md
```

---

## Hardware Requirements & Performance

- **Processor**: Laptop CPU (Intel Core i5/i7 or AMD Ryzen 5/7, 4+ cores). No dedicated GPU required.
- **Memory**: Minimum 8 GB RAM (16 GB recommended for multi-model evaluation).
- **Disk**: ~2 GB for model weights cache (`faster-whisper`, `all-MiniLM-L6-v2`, `nli-deberta-v3-small`).
- **Quantization**: `int8` quantization enabled by default for CPU inference.
- **Execution Speed**: Real-Time Factor (RTF) $\approx 0.15\text{x} - 0.35\text{x}$ (processes a 10-minute meeting in under 2.5 minutes on CPU).

---

## Installation & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Pathan1234-p/meeting-ai-transcription-summary.git
cd meeting-ai-transcription-summary

# Create virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy linguistic model
python -m spacy download en_core_web_sm
```

### 2. Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Key settings in `.env`:
```ini
# Optional Hugging Face token for pyannote.audio diarization
HF_TOKEN=

# Default Whisper ASR size: tiny, base, small, medium
WHISPER_MODEL_SIZE=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Summarizer backend: local (Hugging Face) or api
SUMMARIZER_BACKEND=local
SUMMARIZER_MODEL_NAME=sshleifer/distilbart-cnn-12-6

# NLI Factuality Guard Model
NLI_MODEL_NAME=cross-encoder/nli-deberta-v3-small
```

---

## Usage Guide

### 1. Command-Line Interface (CLI)
Transcribe and analyze any audio file:
```bash
python -m src.pipeline --input meeting.mp3 --out results/ --model-size small
```
With speaker mapping:
```bash
python -m src.pipeline --input meeting.mp3 --out results/ --speaker-map "{\"SPEAKER_00\": \"Anas\", \"SPEAKER_01\": \"Priya\"}"
```
Generated artifacts:
- `results/<meeting_id>_results.json`: Full structured JSON analysis.
- `results/<meeting_id>_report.md`: Formatted Markdown report.

### 2. Interactive Streamlit Web UI
Launch the interactive web application:
```bash
streamlit run app/app.py
```
Open `http://localhost:8501` in your browser. Upload an audio recording, inspect real-time metrics, review dual summaries and hallucination badges, and download results in one click.

---

## Deploy to Cloud

### Deploy to Render

You can deploy this application directly to [Render](https://render.com/) in two ways:

#### Option A: One-Click / Blueprint Deployment (Recommended)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

1. Fork or push this repository to your GitHub account (`Pathan1234-p/meeting-ai-transcription-summary`).
2. Go to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** > **Blueprint**.
4. Select your GitHub repository.
5. Render will automatically read [`render.yaml`](file:///c:/Users/anasp/OneDrive/Desktop/NLP%20project/render.yaml) and configure the web service with:
   - **Runtime**: Python 3.11
   - **Build Command**: `bash build.sh` (installs CPU PyTorch and dependencies)
   - **Start Command**: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`
6. Click **Apply** to deploy!

#### Option B: Manual Web Service Setup
1. On [Render](https://dashboard.render.com/), click **New +** > **Web Service**.
2. Connect your GitHub repository.
3. Configure the following settings:
   - **Name**: `meeting-ai-intelligence`
   - **Runtime**: `Python`
   - **Build Command**: `pip install --upgrade pip && pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command**: `streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0`
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION`: `3.11.4`
   - `WHISPER_MODEL_SIZE`: `tiny` (recommended for cloud free tier)
   - `WHISPER_DEVICE`: `cpu`
   - `WHISPER_COMPUTE_TYPE`: `int8`
5. Click **Deploy Web Service**.


### 3. Run Test Suite
Run all unit and integration tests:
```bash
python -m pytest tests/ -v
```

---

## Evaluation Benchmark Suite

Run the full evaluation across all 5 benchmark meetings with a single command:
```bash
python evaluation/run_all.py
```

This single command executes:
1. **ASR Model Size Benchmark** (`tiny` vs `base` vs `small` on WER, CER, RTF, Speedup).
2. **Summary Quality Evaluation** (ROUGE-1, ROUGE-2, ROUGE-L, BERTScore F1, Compression).
3. **Factual Accuracy & Hallucination Guard** (NLI faithfulness scores, sample sentence verification table).
4. **Completeness & Extraction Evaluation** (Key points coverage, Decisions F1, Action Items F1).
5. **Action Item Attribution Accuracy** (Correct Attribution Rate, Wrong Attribution Rate, Unassigned Recovery).
6. **Granular Error Analysis** (Substitutions/Deletions/Insertions, Top 20 Error Table, Error Propagation).
7. **Ablation Studies** (With/Without Text Cleanup, With/Without Hallucination Guard).
8. **Generates publication-quality charts** (`reports/*.png`).
9. **Compiles full report**: [`reports/evaluation_report.md`](file:///c:/Users/anasp/OneDrive/Desktop/NLP%20project/reports/evaluation_report.md).

---

## Privacy Notice
Audio recordings and generated meeting transcripts contain sensitive discussions. By default, this pipeline runs **100% locally on your CPU** without transmitting audio or text to third-party cloud servers. External cloud APIs are used only if explicitly configured in `.env`.
