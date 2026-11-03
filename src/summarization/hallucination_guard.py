"""
Hallucination Guard module.
Cross-checks each abstractive summary sentence against the meeting transcript
using NLI premise-hypothesis verification to detect unfaithful or fabricated claims.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from src.schemas import SummarySentence
from src.cleanup.cleaner import TextCleaner
from src.config import config

logger = logging.getLogger(__name__)


class HallucinationGuard:
    """Verifies factual faithfulness of generated summaries against meeting transcripts."""

    def __init__(
        self,
        nli_model_name: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        supported_threshold: float = 0.60,
        unsupported_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        self.nli_model_name = nli_model_name or config.nli_model_name
        self.embedding_model_name = embedding_model_name
        self.supported_threshold = supported_threshold
        self.unsupported_threshold = unsupported_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.cleaner = TextCleaner()
        self._embedder = None
        self._nli_tokenizer = None
        self._nli_model = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load sentence embedder for premise candidate retrieval."""
        if self._embedder is None:
            logger.info(f"Loading premise embedder: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _init_nli_model(self):
        """Lazy load NLI classification model and tokenizer."""
        if self._nli_model is None:
            logger.info(f"Loading NLI cross-encoder '{self.nli_model_name}' on {self.device}...")
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_name)
            self._nli_model.to(torch.device(self.device))
            self._nli_model.eval()

    def _get_top_candidate_premises(
        self,
        summary_sentence: str,
        transcript_sentences: List[str],
        top_k: int = 4,
    ) -> List[str]:
        """Retrieve most semantically relevant transcript premises (individual and contiguous pairs)."""
        if not transcript_sentences:
            return []

        # Build candidate pool: single sentences and 2-sentence contiguous context windows
        candidates = list(transcript_sentences)
        for i in range(len(transcript_sentences) - 1):
            candidates.append(f"{transcript_sentences[i]} {transcript_sentences[i+1]}")

        if len(candidates) <= top_k:
            return candidates

        hyp_emb = self.embedder.encode([summary_sentence], convert_to_numpy=True, show_progress_bar=False)
        prem_embs = self.embedder.encode(candidates, convert_to_numpy=True, show_progress_bar=False)

        similarities = np.dot(prem_embs, hyp_emb.T).flatten() / (
            np.linalg.norm(prem_embs, axis=1) * np.linalg.norm(hyp_emb) + 1e-8
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [candidates[idx] for idx in top_indices]

    def _score_nli_pair(self, premise: str, hypothesis: str) -> float:
        """
        Compute entailment probability for a (premise, hypothesis) pair.
        Returns float probability in [0.0, 1.0].
        """
        self._init_nli_model()

        inputs = self._nli_tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._nli_model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        # Hugging Face NLI label mappings
        id2label = getattr(self._nli_model.config, "id2label", {})
        entailment_idx = 1  # Standard for DeBERTa-v3 NLI: {0: contradiction, 1: entailment, 2: neutral}
        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                entailment_idx = int(idx)
                break

        return float(probs[entailment_idx])

    def verify_summary(
        self,
        summary_text: str,
        transcript_text_or_sentences: str | List[str],
    ) -> Tuple[List[SummarySentence], float]:
        """
        Check all sentences of the summary against the source transcript.

        Args:
            summary_text: The abstractive summary text.
            transcript_text_or_sentences: The meeting transcript text or list of sentences.

        Returns:
            Tuple of (list of SummarySentence objects, hallucination_rate)
        """
        if isinstance(transcript_text_or_sentences, str):
            transcript_sentences = self.cleaner.segment_sentences(transcript_text_or_sentences)
        else:
            transcript_sentences = transcript_text_or_sentences

        summary_sentences = self.cleaner.segment_sentences(summary_text)

        if not summary_sentences:
            return [], 0.0

        if not transcript_sentences:
            # No premises to support any claims
            results = [
                SummarySentence(
                    sentence=s,
                    faithfulness_score=0.0,
                    status="Unsupported",
                    supporting_premise=None,
                )
                for s in summary_sentences
            ]
            return results, 1.0

        verified_sentences: List[SummarySentence] = []
        unsupported_count = 0

        for hyp in summary_sentences:
            candidate_premises = self._get_top_candidate_premises(hyp, transcript_sentences, top_k=3)

            best_entailment_score = 0.0
            best_premise = None

            for premise in candidate_premises:
                score = self._score_nli_pair(premise, hyp)
                if score > best_entailment_score:
                    best_entailment_score = score
                    best_premise = premise

            # Classify support status
            if best_entailment_score >= self.supported_threshold:
                status = "Supported"
            elif best_entailment_score >= self.unsupported_threshold:
                status = "Ambiguous"
            else:
                status = "Unsupported"
                unsupported_count += 1

            verified_sentences.append(
                SummarySentence(
                    sentence=hyp,
                    faithfulness_score=round(best_entailment_score, 3),
                    status=status,
                    supporting_premise=best_premise,
                )
            )

        hallucination_rate = round(unsupported_count / len(summary_sentences), 3)
        logger.info(
            f"Hallucination check: {len(summary_sentences)} sentences, "
            f"{unsupported_count} unsupported (rate={hallucination_rate * 100:.1f}%)."
        )

        return verified_sentences, hallucination_rate
