"""
Extractive summarization module combining TextRank graph ranking and
Maximal Marginal Relevance (MMR) clustering via sentence embeddings.
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.cleanup.cleaner import TextCleaner

logger = logging.getLogger(__name__)


class ExtractiveSummarizer:
    """Extracts the most salient and diverse sentences from meeting transcripts."""

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        damping_factor: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-4,
    ):
        self.embedding_model_name = embedding_model_name
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        self._embedder = None
        self.cleaner = TextCleaner()

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy load sentence embedding model."""
        if self._embedder is None:
            logger.info(f"Loading SentenceTransformer: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Compute TextRank centrality scores via power iteration."""
        n = similarity_matrix.shape[0]
        if n == 0:
            return np.array([])
        if n == 1:
            return np.array([1.0])

        # Normalize rows to form transition probability matrix
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        norm_matrix = similarity_matrix / row_sums

        # Power iteration
        scores = np.ones(n) / n
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            scores = (1 - self.damping_factor) / n + self.damping_factor * norm_matrix.T.dot(scores)
            if np.linalg.norm(scores - prev_scores) < self.tol:
                break

        return scores

    def _mmr_selection(
        self,
        embeddings: np.ndarray,
        doc_embedding: np.ndarray,
        k: int,
        lambda_param: float = 0.7,
    ) -> List[int]:
        """
        Maximal Marginal Relevance (MMR) sentence selection.
        Balances query/document relevance against redundancy.
        """
        n = len(embeddings)
        if k >= n:
            return list(range(n))

        # Cosine similarity to doc centroid
        doc_sim = np.dot(embeddings, doc_embedding) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(doc_embedding) + 1e-8
        )

        selected_indices: List[int] = []
        candidate_indices = list(range(n))

        # Select first sentence with highest doc similarity
        first_idx = int(np.argmax(doc_sim))
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)

        # Iteratively select sentences
        while len(selected_indices) < k and candidate_indices:
            best_idx = None
            best_mmr = -float("inf")

            for c_idx in candidate_indices:
                rel = doc_sim[c_idx]

                # Max similarity to already selected sentences
                sim_to_selected = [
                    np.dot(embeddings[c_idx], embeddings[s_idx])
                    / (np.linalg.norm(embeddings[c_idx]) * np.linalg.norm(embeddings[s_idx]) + 1e-8)
                    for s_idx in selected_indices
                ]
                redundancy = max(sim_to_selected) if sim_to_selected else 0.0

                mmr_score = lambda_param * rel - (1.0 - lambda_param) * redundancy
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = c_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)
            else:
                break

        return selected_indices

    def summarize(
        self,
        text_or_sentences: str | List[str],
        ratio: float = 0.20,
        min_sentences: int = 2,
        max_sentences: int = 8,
        method: str = "hybrid",  # 'textrank', 'mmr', or 'hybrid'
    ) -> List[str]:
        """
        Produce extractive summary from meeting transcript.

        Args:
            text_or_sentences: Raw transcript string or list of sentence strings.
            ratio: Fraction of sentences to retain.
            min_sentences: Minimum sentences in output.
            max_sentences: Maximum sentences in output.
            method: Selection method ('textrank', 'mmr', 'hybrid').

        Returns:
            List of selected sentences preserved in original chronological order.
        """
        if isinstance(text_or_sentences, str):
            sentences = self.cleaner.segment_sentences(text_or_sentences)
        else:
            sentences = [s.strip() for s in text_or_sentences if s.strip()]

        if not sentences:
            return []

        n = len(sentences)
        if n <= min_sentences:
            return sentences

        # Compute sentence embeddings
        embeddings = self.embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

        # Number of sentences to select
        k = max(min_sentences, min(int(n * ratio), max_sentences))
        k = min(k, n)

        if method == "textrank":
            # Compute pairwise cosine similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            norm_embeddings = embeddings / norms
            sim_matrix = np.maximum(0.0, np.dot(norm_embeddings, norm_embeddings.T))
            np.fill_diagonal(sim_matrix, 0.0)

            scores = self._textrank(sim_matrix)
            top_indices = np.argsort(scores)[::-1][:k]
            selected_indices = sorted(top_indices.tolist())

        elif method == "mmr":
            doc_embedding = np.mean(embeddings, axis=0)
            selected_indices = sorted(self._mmr_selection(embeddings, doc_embedding, k))

        else:  # hybrid
            # Rank candidates with TextRank, then filter top 2k candidates with MMR
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
            norm_embeddings = embeddings / norms
            sim_matrix = np.maximum(0.0, np.dot(norm_embeddings, norm_embeddings.T))
            np.fill_diagonal(sim_matrix, 0.0)

            scores = self._textrank(sim_matrix)
            candidate_pool_size = min(n, max(k * 2, k))
            pool_indices = np.argsort(scores)[::-1][:candidate_pool_size]

            pool_embeddings = embeddings[pool_indices]
            doc_embedding = np.mean(embeddings, axis=0)
            mmr_sub_indices = self._mmr_selection(pool_embeddings, doc_embedding, k)

            selected_indices = sorted([pool_indices[idx] for idx in mmr_sub_indices])

        summary_sentences = [sentences[idx] for idx in selected_indices]
        return summary_sentences
