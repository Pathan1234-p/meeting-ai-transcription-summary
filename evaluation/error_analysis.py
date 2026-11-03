"""
Transcription Error Analysis and Error Propagation module.
Categorizes ASR errors into:
- Operations: Substitutions, Deletions, Insertions
- Domain categories: Proper nouns/names, technical terms/acronyms, numbers/dates, homophones
- Produces Top 20 errors table with frequencies and examples
- Evaluates downstream error propagation to summaries and action items.
"""

import re
import logging
from collections import Counter
from typing import Dict, Any, List, Tuple
import pandas as pd
import jiwer
from evaluation.asr_eval import normalize_text_for_wer

logger = logging.getLogger("ErrorAnalysis")

# Keyword categories for error classification
TECHNICAL_TERMS = {
    "flashattention", "bf16", "slurm", "h100", "gpus", "gpu", "tls", "pci-dss",
    "pci", "dss", "ssl", "kafka", "rabbitmq", "redis", "ttl", "datadog", "cad",
    "schematics", "lithium-ion", "rds", "aws", "ingress", "oom", "tokenized",
}

NAMES = {"david", "sarah", "michael", "elena", "marcus", "anas", "priya", "bob", "vikram", "maya", "rachel", "kevin"}

NUMBERS_DATES = {
    "twenty-five", "forty", "eight", "three", "five", "seventy-five", "thirty-two",
    "twenty-two", "friday", "monday", "tuesday", "wednesday", "thursday", "tomorrow",
    "25", "40", "75000", "3600", "32", "22",
}

HOMOPHONES = {
    ("sync", "sink"), ("sink", "sync"),
    ("to", "two"), ("two", "to"), ("too", "to"),
    ("there", "their"), ("their", "there"),
    ("for", "four"), ("four", "for"),
    ("hour", "our"), ("our", "hour"),
}


def classify_error_token(ref_word: str, hyp_word: str, op_type: str) -> str:
    """Classify the semantic category of an ASR error."""
    ref_l = ref_word.lower()
    hyp_l = hyp_word.lower()

    if (ref_l, hyp_l) in HOMOPHONES:
        return "Homophone"
    if ref_l in NAMES or hyp_l in NAMES:
        return "Proper Noun / Name"
    if ref_l in TECHNICAL_TERMS or hyp_l in TECHNICAL_TERMS:
        return "Technical Term / Acronym"
    if (
        ref_l in NUMBERS_DATES
        or hyp_l in NUMBERS_DATES
        or any(char.isdigit() for char in ref_word + hyp_word)
    ):
        return "Number / Date / Metric"

    return "General Vocabulary"


class TranscriptionErrorAnalyzer:
    """Performs granular error categorization and propagation analysis."""

    def analyze_transcription_errors(
        self,
        reference_text: str,
        hypothesis_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Analyze token-level alignment between reference and hypothesis using jiwer.
        Extracts substitutions, deletions, and insertions.
        """
        ref_clean = normalize_text_for_wer(reference_text)
        hyp_clean = normalize_text_for_wer(hypothesis_text)

        output = jiwer.process_words(ref_clean, hyp_clean)
        alignments = output.alignments[0]

        ref_words = ref_clean.split()
        hyp_words = hyp_clean.split()

        errors = []
        op_counts = Counter()

        for chunk in alignments:
            op = chunk.type
            op_counts[op] += 1

            if op == "substitute":
                ref_sub = " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx])
                hyp_sub = " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx])
                cat = classify_error_token(ref_sub, hyp_sub, "substitute")
                errors.append({
                    "Operation": "Substitution",
                    "Reference": ref_sub,
                    "Hypothesis": hyp_sub,
                    "Category": cat,
                })
            elif op == "delete":
                ref_del = " ".join(ref_words[chunk.ref_start_idx : chunk.ref_end_idx])
                cat = classify_error_token(ref_del, "", "delete")
                errors.append({
                    "Operation": "Deletion",
                    "Reference": ref_del,
                    "Hypothesis": "(omitted)",
                    "Category": cat,
                })
            elif op == "insert":
                hyp_ins = " ".join(hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx])
                cat = classify_error_token("", hyp_ins, "insert")
                errors.append({
                    "Operation": "Insertion",
                    "Reference": "(none)",
                    "Hypothesis": hyp_ins,
                    "Category": cat,
                })

        return errors, dict(op_counts)

    def generate_top_errors_table(
        self,
        all_errors: List[Dict[str, Any]],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """Aggregate and rank top N most frequent errors."""
        error_tuples = [
            (e["Operation"], e["Reference"], e["Hypothesis"], e["Category"])
            for e in all_errors
        ]
        counts = Counter(error_tuples)

        rows = []
        for (op, ref, hyp, cat), freq in counts.most_common(top_n):
            rows.append({
                "Rank": len(rows) + 1,
                "Error Operation": op,
                "Reference Word(s)": ref,
                "Hypothesis Word(s)": hyp,
                "Category": cat,
                "Frequency": freq,
                "Example Context": f"Ref: '{ref}' -> ASR: '{hyp}'",
            })

        return pd.DataFrame(rows)

    def analyze_error_propagation(
        self,
        asr_errors: List[Dict[str, Any]],
        generated_summary: str,
        extracted_actions: List[Any],
    ) -> List[Dict[str, str]]:
        """
        Examine how specific ASR mistranscriptions propagate into summary
        misrepresentations or corrupted action items.
        """
        propagation_cases = []

        summary_lower = generated_summary.lower()
        actions_str = " ".join(f"{a.owner} {a.description} {a.deadline}" for a in extracted_actions).lower()

        for err in asr_errors:
            ref = err["Reference"].lower()
            hyp = err["Hypothesis"].lower()

            if err["Operation"] == "Substitution" and len(ref) >= 3 and len(hyp) >= 3:
                # Check if the erroneous word propagated into summary or action items
                if hyp in summary_lower or hyp in actions_str:
                    propagation_cases.append({
                        "Type": f"ASR -> Downstream ({err['Category']})",
                        "ASR Mistranscription": f"'{ref}' misrecognized as '{hyp}'",
                        "Impacted Artifact": "Summary / Action Items",
                        "Propagation Effect": f"Downstream generated text inherited '{hyp}' instead of ground truth '{ref}'.",
                    })

        return propagation_cases[:5]  # Top representative examples
