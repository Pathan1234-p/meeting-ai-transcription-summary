"""
Completeness & Extraction Evaluation module.
Evaluates:
- Key points checklist coverage (Captured / Partially Captured / Missed)
- Precision, Recall, F1 for Decisions
- Precision, Recall, F1 for Action Items
- Action Item Attribution Accuracy, Wrongly Assigned Owner Rate, and Unassigned Recovery Rate.
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("CompletenessEval")


class CompletenessEvaluator:
    """Evaluates coverage of key points, decision extraction, and action item ownership attribution."""

    def __init__(self, embedder_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_name)

    def evaluate_key_points_coverage(
        self,
        key_points: List[str],
        generated_summary: str,
    ) -> Dict[str, Any]:
        """
        Checklist-based key point coverage evaluation.
        Scores each reference key point as Captured (sim >= 0.65),
        Partially Captured (0.45 <= sim < 0.65), or Missed (sim < 0.45).
        """
        if not key_points:
            return {"coverage_rate": 1.0, "captured": 0, "partial": 0, "missed": 0, "breakdown": []}

        sentences = [s.strip() for s in generated_summary.split(".") if s.strip()]
        if not sentences:
            return {
                "coverage_rate": 0.0,
                "captured": 0,
                "partial": 0,
                "missed": len(key_points),
                "breakdown": [{"point": p, "status": "Missed", "score": 0.0} for p in key_points],
            }

        kp_embs = self.embedder.encode(key_points, convert_to_numpy=True, show_progress_bar=False)
        sent_embs = self.embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        sims = np.dot(kp_embs, sent_embs.T) / (
            np.linalg.norm(kp_embs, axis=1, keepdims=True) * np.linalg.norm(sent_embs, axis=1, keepdims=True).T + 1e-8
        )
        max_sims = np.max(sims, axis=1)

        captured = 0
        partial = 0
        missed = 0
        breakdown = []

        for kp, sim in zip(key_points, max_sims):
            if sim >= 0.65:
                status = "Captured"
                captured += 1
            elif sim >= 0.45:
                status = "Partially Captured"
                partial += 1
            else:
                status = "Missed"
                missed += 1

            breakdown.append({"point": kp, "status": status, "score": round(float(sim), 3)})

        total = len(key_points)
        coverage_rate = round((captured + 0.5 * partial) / total, 3)

        return {
            "coverage_rate": coverage_rate,
            "captured": captured,
            "partial": partial,
            "missed": missed,
            "total_points": total,
            "breakdown": breakdown,
        }

    def evaluate_decisions(
        self,
        extracted_decisions: List[Any],
        reference_decisions: List[Dict[str, Any]],
        sim_threshold: float = 0.65,
    ) -> Dict[str, float]:
        """Compute Precision, Recall, and F1 for decision extraction."""
        n_ext = len(extracted_decisions)
        n_ref = len(reference_decisions)

        if n_ref == 0 and n_ext == 0:
            return {"Precision": 100.0, "Recall": 100.0, "F1": 100.0}
        if n_ref == 0 or n_ext == 0:
            return {"Precision": 0.0, "Recall": 0.0, "F1": 0.0}

        ext_texts = [d.statement for d in extracted_decisions]
        ref_texts = [d["statement"] for d in reference_decisions]

        ext_embs = self.embedder.encode(ext_texts, convert_to_numpy=True, show_progress_bar=False)
        ref_embs = self.embedder.encode(ref_texts, convert_to_numpy=True, show_progress_bar=False)

        sim_matrix = np.dot(ext_embs, ref_embs.T) / (
            np.linalg.norm(ext_embs, axis=1, keepdims=True) * np.linalg.norm(ref_embs, axis=1, keepdims=True).T + 1e-8
        )

        matched_ref = set()
        tp = 0
        for i in range(n_ext):
            best_j = int(np.argmax(sim_matrix[i]))
            if sim_matrix[i, best_j] >= sim_threshold:
                tp += 1
                matched_ref.add(best_j)

        precision = (tp / n_ext) if n_ext > 0 else 0.0
        recall = (len(matched_ref) / n_ref) if n_ref > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "Precision": round(precision * 100, 1),
            "Recall": round(recall * 100, 1),
            "F1": round(f1 * 100, 1),
        }

    def evaluate_action_items(
        self,
        extracted_actions: List[Any],
        reference_actions: List[Dict[str, Any]],
        sim_threshold: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Evaluate Action Item extraction and rigorous Owner Attribution Accuracy:
        - Task extraction Precision, Recall, F1
        - Correct Owner Attribution Rate (for matched items)
        - Wrongly Assigned Owner Rate (severely penalized)
        - Unassigned Recovery Rate
        """
        n_ext = len(extracted_actions)
        n_ref = len(reference_actions)

        if n_ref == 0 and n_ext == 0:
            return {
                "Precision": 100.0, "Recall": 100.0, "F1": 100.0,
                "Attribution Accuracy (%)": 100.0, "Wrong Attribution Rate (%)": 0.0,
                "Unassigned Recovery (%)": 100.0,
            }
        if n_ref == 0 or n_ext == 0:
            return {
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "Attribution Accuracy (%)": 0.0, "Wrong Attribution Rate (%)": 0.0,
                "Unassigned Recovery (%)": 0.0,
            }

        ext_texts = [a.description for a in extracted_actions]
        ref_texts = [a["description"] for a in reference_actions]

        ext_embs = self.embedder.encode(ext_texts, convert_to_numpy=True, show_progress_bar=False)
        ref_embs = self.embedder.encode(ref_texts, convert_to_numpy=True, show_progress_bar=False)

        sim_matrix = np.dot(ext_embs, ref_embs.T) / (
            np.linalg.norm(ext_embs, axis=1, keepdims=True) * np.linalg.norm(ref_embs, axis=1, keepdims=True).T + 1e-8
        )

        tp = 0
        correct_owner = 0
        wrong_owner = 0
        correct_unassigned = 0
        total_ref_unassigned = sum(1 for a in reference_actions if a.get("owner", "").lower() == "unassigned")

        matched_ref_indices = set()

        for i in range(n_ext):
            best_j = int(np.argmax(sim_matrix[i]))
            if sim_matrix[i, best_j] >= sim_threshold:
                tp += 1
                matched_ref_indices.add(best_j)

                ext_owner = extracted_actions[i].owner.strip().lower()
                ref_owner = reference_actions[best_j].get("owner", "unassigned").strip().lower()

                if ext_owner == ref_owner:
                    correct_owner += 1
                    if ref_owner == "unassigned":
                        correct_unassigned += 1
                else:
                    # If model assigned a real name, but reference was different or unassigned:
                    if ext_owner != "unassigned":
                        wrong_owner += 1

        precision = (tp / n_ext) if n_ext > 0 else 0.0
        recall = (len(matched_ref_indices) / n_ref) if n_ref > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        attribution_acc = (correct_owner / max(tp, 1)) * 100
        wrong_attribution_rate = (wrong_owner / max(tp, 1)) * 100
        unassigned_recovery = (correct_unassigned / max(total_ref_unassigned, 1)) * 100

        return {
            "Precision": round(precision * 100, 1),
            "Recall": round(recall * 100, 1),
            "F1": round(f1 * 100, 1),
            "Attribution Accuracy (%)": round(attribution_acc, 1),
            "Wrong Attribution Rate (%)": round(wrong_attribution_rate, 1),
            "Unassigned Recovery (%)": round(unassigned_recovery, 1),
            "Total Matched": tp,
            "Correct Owners": correct_owner,
            "Wrong Owners": wrong_owner,
        }
