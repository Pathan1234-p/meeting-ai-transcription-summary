"""
AI-Powered Meeting Transcription & Summary Generation - Streamlit Web Application.
Provides interactive audio upload, playback, speaker-attributed transcripts,
extractive vs abstractive summaries, hallucination guard badges, decisions log,
action items table with strict explicit attribution, and export options.
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.pipeline import MeetingPipeline
from src.schemas import MeetingAnalysisResult
from src.config import config

st.set_page_config(
    page_title="MeetingAI - Transcription & Summary Intelligence",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .speaker-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #3b82f6;
        color: #ffffff;
        margin-right: 6px;
    }
    .status-supported {
        color: #10b981;
        font-weight: 600;
    }
    .status-ambiguous {
        color: #f59e0b;
        font-weight: 600;
    }
    .status-unsupported {
        color: #ef4444;
        font-weight: 600;
    }
    .owner-badge-assigned {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    .owner-badge-unassigned {
        background-color: #78350f;
        color: #fde68a;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown("<div class='main-header'>🎙️ MeetingAI Intelligence Suite</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Clean Transcripts • Dual Extractive & Abstractive Summaries • NLI Hallucination Guard • Explicit Action Items</div>",
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        model_size = st.selectbox(
            "Whisper ASR Model Size",
            options=["tiny", "base", "small", "medium"],
            index=0,  # Default tiny for snappy local demonstration
            help="Tiny/Base for fast CPU execution; Small/Medium for higher transcription accuracy.",
        )
        hf_token = st.text_input(
            "Hugging Face Token (Optional)",
            value="",
            type="password",
            help="Required only for pyannote.audio diarization. If blank, falls back gracefully.",
        )
        speaker_mapping_input = st.text_area(
            "Speaker Name Mapping (JSON)",
            value='{\n  "SPEAKER_00": "Anas",\n  "SPEAKER_01": "Priya",\n  "SPEAKER_02": "Bob"\n}',
            help="Rename generic diarization speaker labels to real attendee names.",
        )

        st.markdown("---")
        st.markdown("### 🔒 Privacy Notice")
        st.info("All audio processing, transcription, summarization, and hallucination checks run 100% locally on your CPU. No audio leaves this machine.")

    # Main Area: Audio Input Selection
    col_input1, col_input2 = st.columns([1, 1])

    with col_input1:
        uploaded_file = st.file_uploader(
            "Upload Meeting Recording (MP3, WAV, M4A, MP4)",
            type=["wav", "mp3", "m4a", "flac", "ogg", "mp4"],
        )

    with col_input2:
        # Check for demo/sample audio files
        preset_files = []
        demo_speech = PROJECT_ROOT / "tests" / "demo_speech.wav"
        if demo_speech.exists():
            preset_files.append("Sample Meeting Sync (demo_speech.wav)")

        preset_selection = st.selectbox(
            "Or Choose a Benchmark Sample",
            options=["(None)"] + preset_files,
        )

    # Determine audio source
    active_audio_path = None
    if uploaded_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        active_audio_path = temp_path
    elif preset_selection != "(None)":
        active_audio_path = str(demo_speech)

    if active_audio_path:
        st.audio(active_audio_path)

        # Process Button
        if st.button("🚀 Process Meeting Documentation", type="primary"):
            speaker_map = None
            if speaker_mapping_input.strip():
                try:
                    speaker_map = json.loads(speaker_mapping_input)
                except Exception as e:
                    st.warning(f"Could not parse speaker mapping JSON: {e}")

            with st.spinner(f"Running Meeting AI pipeline with Whisper ({model_size})..."):
                pipeline = MeetingPipeline(
                    whisper_model_size=model_size,
                    hf_token=hf_token,
                )
                result = pipeline.process(
                    audio_path=active_audio_path,
                    speaker_mapping=speaker_map,
                )
                st.session_state["meeting_result"] = result

    # Display Results if available
    if "meeting_result" in st.session_state:
        result: MeetingAnalysisResult = st.session_state["meeting_result"]

        st.markdown("---")
        st.subheader("📊 Meeting Overview & Diagnostics")

        # Diagnostics metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Duration", f"{result.transcript.duration_seconds:.1f}s")
        with m2:
            st.metric("Total Words", result.transcript.word_count)
        with m3:
            st.metric("Real-Time Factor", f"{result.metadata.get('real_time_factor', 0):.2f}x")
        with m4:
            st.metric("Decisions", len(result.decisions))
        with m5:
            st.metric("Action Items", len(result.action_items))

        # Main tabs
        tab_sum, tab_actions, tab_decisions, tab_transcript, tab_export = st.tabs([
            "📝 Summaries & Faithfulness",
            "🎯 Action Items",
            "⚖️ Decisions Log",
            "🗣️ Speaker Transcript",
            "💾 Export Reports",
        ])

        # TAB 1: Summaries
        with tab_sum:
            st.markdown("### Executive Highlights")
            for b in result.summary.executive_bullets:
                st.markdown(f"• **{b}**")

            st.markdown("---")
            c_abs, c_ext = st.columns(2)

            with c_abs:
                st.markdown("#### 🤖 Abstractive Summary (with Hallucination Guard)")
                st.write(result.summary.abstractive_summary)

                st.markdown(f"**Hallucination Rate**: `{result.summary.hallucination_rate * 100:.1f}%`")
                if result.summary.abstractive_sentences:
                    with st.expander("🔍 Sentence-Level NLI Faithfulness Breakdown", expanded=True):
                        for s in result.summary.abstractive_sentences:
                            if s.status == "Supported":
                                badge = "🟢 **Supported**"
                            elif s.status == "Ambiguous":
                                badge = "🟡 **Ambiguous**"
                            else:
                                badge = "🔴 **Unsupported (Hallucination)**"

                            st.markdown(f"{badge} (Score: `{s.faithfulness_score:.2f}`): *\"{s.sentence}\"*")
                            if s.supporting_premise:
                                st.caption(f"↳ Supporting Premise: *\"{s.supporting_premise}\"*")

            with c_ext:
                st.markdown("#### 📌 Extractive Key Points (TextRank + MMR)")
                for sentence in result.summary.extractive_summary:
                    st.info(sentence)
                st.caption(f"Compression Ratio: `{result.summary.compression_ratio * 100:.1f}%` of original transcript length.")

        # TAB 2: Action Items
        with tab_actions:
            st.markdown("### 🎯 Extracted Action Items")
            st.caption("Strict Explicit Ownership: Assigned ONLY when explicitly stated in transcript. Ambiguous items are tagged Unassigned. Never guessed.")

            if result.action_items:
                act_data = []
                for a in result.action_items:
                    owner_pill = f"🟢 {a.owner}" if a.owner != "Unassigned" else "🟡 Unassigned"
                    act_data.append({
                        "#": a.action_id,
                        "Task Description": a.description,
                        "Owner": owner_pill,
                        "Deadline": a.deadline or "-",
                        "Speaker": a.source_speaker,
                        "Timestamp": f"{a.timestamp:.1f}s",
                        "Supporting Quote": a.supporting_quote,
                        "Confidence": f"{a.confidence:.2f}",
                    })
                st.dataframe(pd.DataFrame(act_data), use_container_width=True, hide_index=True)
            else:
                st.info("No action items detected in this meeting.")

        # TAB 3: Decisions
        with tab_decisions:
            st.markdown("### ⚖️ Formal Decisions & Consensus")
            if result.decisions:
                dec_data = []
                for d in result.decisions:
                    dec_data.append({
                        "#": d.decision_id,
                        "Decision Statement": d.statement,
                        "Proposed / Confirmed By": d.proposed_by or "Consensus",
                        "Timestamp": f"{d.timestamp:.1f}s",
                        "Supporting Quote": d.supporting_quote,
                        "Confidence": f"{d.confidence:.2f}",
                    })
                st.dataframe(pd.DataFrame(dec_data), use_container_width=True, hide_index=True)
            else:
                st.info("No formal decisions identified in this recording.")

        # TAB 4: Speaker Transcript
        with tab_transcript:
            st.markdown("### 🗣️ Speaker-Attributed Transcript")
            for seg in result.transcript.segments:
                speaker_color = "#3b82f6"
                st.markdown(
                    f"<span style='color: {speaker_color}; font-weight: bold;'>[{seg.start:.1f}s - {seg.end:.1f}s] {seg.speaker}</span>: {seg.text}",
                    unsafe_allow_html=True,
                )

        # TAB 5: Export
        with tab_export:
            st.markdown("### 💾 Export Structured Results")
            json_str = result.model_dump_json(indent=2)

            # Generate markdown text
            md_lines = [
                f"# Meeting Documentation: {result.meeting_id}",
                f"**Duration**: {result.transcript.duration_seconds:.1f}s | **Words**: {result.transcript.word_count}",
                "\n## Executive Summary",
            ]
            for b in result.summary.executive_bullets:
                md_lines.append(f"- {b}")
            md_lines.append(f"\n## Abstractive Summary\n{result.summary.abstractive_summary}")
            md_lines.append(f"\n## Decisions")
            for d in result.decisions:
                md_lines.append(f"- [{d.timestamp:.1f}s] {d.statement} (Proposed by: {d.proposed_by or 'Consensus'})")
            md_lines.append(f"\n## Action Items")
            for a in result.action_items:
                md_lines.append(f"- [Owner: {a.owner} | Due: {a.deadline or 'None'}] {a.description}")
            md_str = "\n".join(md_lines)

            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    label="📥 Download Structured JSON",
                    data=json_str,
                    file_name=f"{result.meeting_id}_results.json",
                    mime="application/json",
                )
            with c_dl2:
                st.download_button(
                    label="📥 Download Markdown Report",
                    data=md_str,
                    file_name=f"{result.meeting_id}_report.md",
                    mime="text/markdown",
                )


if __name__ == "__main__":
    main()
