"""
Abstractive summarization module supporting local Hugging Face transformer models
(e.g. distilbart-cnn-12-6, bart-large-cnn) with hierarchical map-reduce for long transcripts,
and pluggable API backend.
"""

import os
import logging
from typing import List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.config import config

logger = logging.getLogger(__name__)


class AbstractiveSummarizer:
    """Produces fluent abstractive meeting summaries and executive bullet points."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        max_chunk_words: int = 500,
        device: Optional[str] = None,
    ):
        self.backend = backend or config.summarizer_backend
        self.model_name = model_name or config.summarizer_model_name
        self.max_chunk_words = max_chunk_words
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._tokenizer = None

    def _init_local_pipeline(self):
        """Lazy-load the seq2seq model and tokenizer directly.

        transformers v5 removed both 'summarization' and 'text2text-generation'
        from the pipeline() API. We use AutoModelForSeq2SeqLM.generate() +
        AutoTokenizer directly, which works across all transformers versions.
        """
        if self._model is None:
            logger.info(f"Loading abstractive model '{self.model_name}' on {self.device}...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model = self._model.to(self.device)
            self._model.eval()

    def _chunk_text_by_words(self, text: str, max_words: int) -> List[str]:
        """Split text into manageable chunks respecting sentence boundaries where possible."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        current_chunk = []
        current_count = 0

        for word in words:
            current_chunk.append(word)
            current_count += 1
            if current_count >= max_words and word.endswith((".", "!", "?")):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _summarize_chunk(self, chunk_text: str, max_length: int = 150, min_length: int = 40) -> str:
        """Summarize a single chunk with local HF model or API."""
        if self.backend == "api":
            return self._summarize_chunk_api(chunk_text, max_length)

        self._init_local_pipeline()
        input_len = len(chunk_text.split())
        # Adjust length bounds to prevent Hugging Face warnings
        effective_max = min(max_length, max(30, int(input_len * 0.8)))
        effective_min = min(min_length, max(10, int(input_len * 0.3)))

        try:
            self._init_local_pipeline()
            inputs = self._tokenizer(
                chunk_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self.device)
            summary_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=effective_max,
                min_new_tokens=effective_min,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            decoded = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return decoded.strip()
        except Exception as e:
            logger.error(f"Local summarization error: {e}")
            # Fallback: return leading sentences
            return ". ".join(chunk_text.split(". ")[:2]) + "."

    def _summarize_chunk_api(self, chunk_text: str, max_length: int) -> str:
        """Optional API-based summarization (e.g. OpenAI)."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Falling back to local pipeline.")
            self.backend = "local"
            return self._summarize_chunk(chunk_text, max_length)

        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior executive assistant. Summarize the following meeting transcript accurately and concisely.",
                    },
                    {"role": "user", "content": chunk_text},
                ],
                max_tokens=max_length,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"API summarization failed ({e}). Falling back to local pipeline.")
            self.backend = "local"
            return self._summarize_chunk(chunk_text, max_length)

    def summarize(
        self,
        transcript_text: str,
        max_length: int = 250,
        min_length: int = 50,
    ) -> Tuple[str, List[str]]:
        """
        Generate hierarchical abstractive summary and executive bullet points.

        Args:
            transcript_text: Cleaned transcript text.
            max_length: Target max tokens in summary.
            min_length: Target min tokens in summary.

        Returns:
            Tuple of (abstractive_summary_paragraph, list_of_executive_bullets)
        """
        words = transcript_text.split()
        if len(words) < 25:
            # Too short for abstractive model, return as-is
            return transcript_text, [f"Discussion: {transcript_text}"]

        chunks = self._chunk_text_by_words(transcript_text, self.max_chunk_words)

        if len(chunks) == 1:
            abstractive_summary = self._summarize_chunk(chunks[0], max_length, min_length)
        else:
            # Map step: summarize each chunk
            logger.info(f"Hierarchical summarization over {len(chunks)} chunks...")
            intermediate_summaries = []
            for i, chunk in enumerate(chunks):
                inter_sum = self._summarize_chunk(chunk, max_length=100, min_length=25)
                intermediate_summaries.append(inter_sum)

            # Reduce step: summarize combined intermediate summaries
            combined_intermediate = " ".join(intermediate_summaries)
            abstractive_summary = self._summarize_chunk(
                combined_intermediate, max_length=max_length, min_length=min_length
            )

        # Generate executive bullet points
        executive_bullets = self._generate_executive_bullets(abstractive_summary)

        return abstractive_summary, executive_bullets

    def _generate_executive_bullets(self, summary_text: str) -> List[str]:
        """Convert abstractive summary sentences into crisp executive bullet points."""
        raw_sentences = [s.strip() for s in summary_text.replace("\n", " ").split(".") if s.strip()]
        bullets = []
        for sent in raw_sentences:
            if len(sent.split()) >= 4:
                # Add bullet symbol formatting
                bullets.append(f"{sent.capitalize()}.")

        if not bullets:
            bullets = [summary_text]

        return bullets[:5]
