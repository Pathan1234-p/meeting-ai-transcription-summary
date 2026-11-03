"""
Factuality & Hallucination Evaluation module.
Evaluates NLI-based factual faithfulness of abstractive summaries against source transcripts,
generates classification tables (Supported / Ambiguous / Unsupported), and computes hallucination rates.
"""

import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
from src.summarization.hallucination_guard import HallucinationGuard

logger = logging.getLogger("FactualityEval")


class FactualityEvaluator:
    """Evaluates factual consistency and hallucination rates in generated summaries."""

    def __init__(self):
        self.guard = HallucinationGuard()

    def evaluate_meetings(
        self,
        meetings_data: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """
        Evaluate faithfulness across benchmark meetings.

        Args:
            meetings_data: List of dicts containing 'name', 'abstractive_summary', 'transcript_text'.

        Returns:
            Tuple of:
            - summary_df: Summary metrics per meeting (total sentences, supported, ambiguous, unsupported, rate)
            - sentence_table_df: Sample sentence breakdown with premise, hypothesis, score, and classification
            - overall_hallucination_rate: float
        """
        summary_rows = []
        sentence_rows = []
        total_sentences_all = 0
        total_unsupported_all = 0

        for m in meetings_data:
            meeting_name = m["name"]
            summary_text = m["abstractive_summary"]
            transcript_text = m["transcript_text"]

            verified_sentences, hall_rate = self.guard.verify_summary(summary_text, transcript_text)

            n_total = len(verified_sentences)
            n_supp = sum(1 for s in verified_sentences if s.status == "Supported")
            n_amb = sum(1 for s in verified_sentences if s.status == "Ambiguous")
            n_unsupp = sum(1 for s in verified_sentences if s.status == "Unsupported")

            total_sentences_all += n_total
            total_unsupported_all += n_unsupp

            summary_rows.append({
                "Meeting": meeting_name,
                "Summary Sentences": n_total,
                "Supported (Entailment)": n_supp,
                "Ambiguous (Neutral)": n_amb,
                "Unsupported (Contradiction/Hallucinated)": n_unsupp,
                "Faithfulness Rate (%)": round((n_supp / max(n_total, 1)) * 100, 1),
                "Hallucination Rate (%)": round(hall_rate * 100, 1),
            })

            for s in verified_sentences:
                sentence_rows.append({
                    "Meeting": meeting_name,
                    "Summary Sentence": s.sentence,
                    "Supporting Premise": s.supporting_premise or "(None)",
                    "Faithfulness Score": s.faithfulness_score,
                    "Classification": s.status,
                })

        overall_hall_rate = round(total_unsupported_all / max(total_sentences_all, 1), 3)

        return pd.DataFrame(summary_rows), pd.DataFrame(sentence_rows), overall_hall_rate
