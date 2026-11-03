"""
Unit tests for ExtractiveSummarizer and HallucinationGuard.
"""

import pytest
from src.summarization.extractive import ExtractiveSummarizer
from src.summarization.hallucination_guard import HallucinationGuard


@pytest.fixture(scope="module")
def extractive_summarizer():
    return ExtractiveSummarizer()


@pytest.fixture(scope="module")
def hallucination_guard():
    return HallucinationGuard()


def test_extractive_summarizer_basic(extractive_summarizer):
    """Test extractive summary produces concise, chronological output."""
    sentences = [
        "Welcome to the quarterly budget and roadmap planning session.",
        "Revenue grew by fifteen percent compared to last quarter.",
        "However, infrastructure cloud costs increased significantly.",
        "To mitigate this, Alice suggested migrating non-critical services to ARM instances.",
        "Bob agreed to evaluate ARM instances and benchmark the latency by next Tuesday.",
        "In addition, we need to finalize the hiring plan for two backend engineers.",
        "Priya will review the job descriptions before Friday.",
        "Let's reconvene next week to review the benchmark results.",
    ]

    summary = extractive_summarizer.summarize(sentences, ratio=0.4, min_sentences=2, max_sentences=4)
    assert 2 <= len(summary) <= 4

    # Ensure selected sentences preserve original chronological ordering
    indices = [sentences.index(s) for s in summary]
    assert indices == sorted(indices)


def test_extractive_summarizer_short_input(extractive_summarizer):
    """Test graceful handling of very short text."""
    short = ["Just one sentence here."]
    summary = extractive_summarizer.summarize(short, min_sentences=2)
    assert len(summary) == 1
    assert summary[0] == "Just one sentence here."


def test_hallucination_guard_verification(hallucination_guard):
    """Test that faithful sentences are Supported and fabricated sentences are Unsupported."""
    transcript = (
        "Alice confirmed that the team will migrate the backend database to PostgreSQL. "
        "The migration is scheduled for Friday evening to minimize user disruption. "
        "Bob will monitor the replication lag."
    )

    # 1. Supported claim
    faithful_summary = "The team will migrate the database to PostgreSQL on Friday."
    results, hall_rate = hallucination_guard.verify_summary(faithful_summary, transcript)
    assert len(results) == 1
    assert results[0].status in ["Supported", "Ambiguous"]
    assert results[0].faithfulness_score > 0.35

    # 2. Fabricated claim (hallucination)
    fabricated_summary = "The company bought a fleet of fifty airplanes for international travel."
    results_fab, hall_rate_fab = hallucination_guard.verify_summary(fabricated_summary, transcript)
    assert len(results_fab) == 1
    assert results_fab[0].status == "Unsupported"
    assert results_fab[0].faithfulness_score < 0.35
    assert hall_rate_fab == 1.0
