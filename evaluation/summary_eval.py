"""
Summary Evaluation module.
Evaluates summary quality against human reference summaries using:
- ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1)
- BERTScore (Precision, Recall, F1)
- Compression ratio
- Comparative analysis between Extractive and Abstractive approaches.
"""

import logging
from typing import Dict, Any, List
import pandas as pd
from rouge_score import rouge_scorer
import bert_score

logger = logging.getLogger("SummaryEval")


class SummaryEvaluator:
    """Evaluates generated meeting summaries against reference summaries."""

    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def evaluate_summary(
        self,
        hypothesis_summary: str,
        reference_summary: str,
        transcript_text: str,
    ) -> Dict[str, float]:
        """Compute ROUGE, BERTScore, and compression ratio for a single summary."""
        # 1. ROUGE scores
        scores = self.rouge.score(reference_summary, hypothesis_summary)

        # 2. BERTScore
        # Run BERTScore with distilbert-base-uncased for fast CPU evaluation
        P, R, F1 = bert_score.score(
            [hypothesis_summary],
            [reference_summary],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )

        # 3. Compression ratio
        hyp_len = len(hypothesis_summary.split())
        trans_len = max(len(transcript_text.split()), 1)
        compression_ratio = hyp_len / trans_len

        return {
            "ROUGE-1 F1": round(scores["rouge1"].fmeasure * 100, 2),
            "ROUGE-1 Recall": round(scores["rouge1"].recall * 100, 2),
            "ROUGE-1 Precision": round(scores["rouge1"].precision * 100, 2),
            "ROUGE-2 F1": round(scores["rouge2"].fmeasure * 100, 2),
            "ROUGE-2 Recall": round(scores["rouge2"].recall * 100, 2),
            "ROUGE-2 Precision": round(scores["rouge2"].precision * 100, 2),
            "ROUGE-L F1": round(scores["rougeL"].fmeasure * 100, 2),
            "ROUGE-L Recall": round(scores["rougeL"].recall * 100, 2),
            "ROUGE-L Precision": round(scores["rougeL"].precision * 100, 2),
            "BERTScore F1": round(F1.item() * 100, 2),
            "BERTScore Precision": round(P.item() * 100, 2),
            "BERTScore Recall": round(R.item() * 100, 2),
            "Compression Ratio (%)": round(compression_ratio * 100, 2),
            "Summary Words": hyp_len,
        }

    def compare_extractive_vs_abstractive(
        self,
        meetings_results: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Generate side-by-side comparison table between extractive and abstractive summaries."""
        rows = []
        for m in meetings_results:
            meeting_name = m["name"]
            ref_sum = m["reference_summary"]
            trans_text = m["transcript_text"]
            ext_sum = " ".join(m["extractive_summary"])
            abs_sum = m["abstractive_summary"]

            ext_metrics = self.evaluate_summary(ext_sum, ref_sum, trans_text)
            abs_metrics = self.evaluate_summary(abs_sum, ref_sum, trans_text)

            rows.append({
                "Meeting": meeting_name,
                "Type": "Extractive (TextRank+MMR)",
                "ROUGE-1 F1": ext_metrics["ROUGE-1 F1"],
                "ROUGE-2 F1": ext_metrics["ROUGE-2 F1"],
                "ROUGE-L F1": ext_metrics["ROUGE-L F1"],
                "BERTScore F1": ext_metrics["BERTScore F1"],
                "Compression (%)": ext_metrics["Compression Ratio (%)"],
            })
            rows.append({
                "Meeting": meeting_name,
                "Type": "Abstractive (Chunked Seq2Seq)",
                "ROUGE-1 F1": abs_metrics["ROUGE-1 F1"],
                "ROUGE-2 F1": abs_metrics["ROUGE-2 F1"],
                "ROUGE-L F1": abs_metrics["ROUGE-L F1"],
                "BERTScore F1": abs_metrics["BERTScore F1"],
                "Compression (%)": abs_metrics["Compression Ratio (%)"],
            })

        return pd.DataFrame(rows)
