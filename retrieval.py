from __future__ import annotations

import time
from statistics import median
from typing import Callable, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "NovaSearch/stella_en_1.5B_v5"
# Verified Hugging Face commit; pin it so custom remote code and weights do not drift.
STELLA_REVISION = "7817065102fd9e1b031fe874e910c01f40b2f001"


def choose_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


class StellaRetriever:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        revision: str = STELLA_REVISION,
        device: str | None = None,
        max_seq_length: int = 512,
        batch_size: int = 4,
        use_fp16: bool = True,
    ):
        self.device = choose_device(device)
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.use_fp16 = bool(use_fp16 and self.device.startswith("cuda"))

        precision = "fp16" if self.use_fp16 else "fp32"
        print(
            f"Loading Stella on: {self.device} "
            f"({precision}, max_seq_length={max_seq_length}, "
            f"batch_size={batch_size})"
        )

        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            trust_remote_code=True,
            device=self.device,
        )
        self.model.max_seq_length = max_seq_length

        if self.use_fp16:
            self.model.half()

        # SentenceTransformer exposes tokenizer on current versions; keep a
        # fallback for compatibility with 3.0.1.
        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = self.model._first_module().tokenizer
        self.tokenizer.model_max_length = max_seq_length

    def encode_memories(
        self,
        memories: list[dict[str, Any]],
    ) -> np.ndarray:
        texts = [str(memory["text"]) for memory in memories]
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query,
            prompt_name="s2p_query",
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding)

    def semantic_cosine_scores(
        self,
        query: str,
        memory_embeddings: np.ndarray,
    ) -> np.ndarray:
        query_embedding = self.encode_query(query)
        return memory_embeddings @ query_embedding

    def score_query(
        self,
        query: str,
        memories: list[dict[str, Any]],
        memory_embeddings: np.ndarray,
    ) -> dict[str, np.ndarray | bool]:
        """Compute every reusable per-question retrieval score once."""
        cosine_scores = self.semantic_cosine_scores(query, memory_embeddings)
        if not np.isfinite(cosine_scores).all():
            raise FloatingPointError("Stella produced non-finite semantic scores")
        recency_scores = calculate_recency_scores(memories)
        semantic_scaled = (cosine_scores + 1.0) / 2.0
        semantic_minmax, semantic_constant = minmax_normalize(cosine_scores)
        recency_minmax, recency_constant = minmax_normalize(recency_scores)
        semantic_ranks = stable_ranks(cosine_scores, memories)
        recency_ranks = stable_ranks(recency_scores, memories)
        return {
            "cosine": cosine_scores,
            "semantic_scaled": semantic_scaled,
            "recency": recency_scores,
            "hybrid_raw": 0.5 * semantic_scaled + 0.5 * recency_scores,
            "hybrid_minmax": 0.5 * semantic_minmax + 0.5 * recency_minmax,
            "hybrid_rrf": 1.0 / (60.0 + semantic_ranks) + 1.0 / (60.0 + recency_ranks),
            "semantic_minmax_constant": semantic_constant,
            "recency_minmax_constant": recency_constant,
        }

    def token_length_diagnostics(
        self,
        memories: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Count pre-truncation Stella tokens to quantify truncation risk."""
        lengths: list[int] = []
        for memory in memories:
            ids = self.tokenizer.encode(
                str(memory["text"]),
                add_special_tokens=True,
                truncation=False,
            )
            lengths.append(len(ids))

        if not lengths:
            return {
                "mean_memory_tokens_stella": 0.0,
                "max_memory_tokens_stella": 0,
                "num_memories_over_stella_limit": 0,
                "fraction_memories_over_stella_limit": 0.0,
            }

        over = sum(x > self.max_seq_length for x in lengths)
        return {
            "mean_memory_tokens_stella": float(np.mean(lengths)),
            "max_memory_tokens_stella": int(max(lengths)),
            "num_memories_over_stella_limit": int(over),
            "fraction_memories_over_stella_limit": float(over / len(lengths)),
        }

    def hybrid_component_diagnostics(
        self,
        query: str,
        memories: list[dict[str, Any]],
        memory_embeddings: np.ndarray,
        score_bundle: dict[str, np.ndarray | bool] | None = None,
    ) -> dict[str, float]:
        if score_bundle is None:
            score_bundle = self.score_query(query, memories, memory_embeddings)
        semantic_scores = np.asarray(score_bundle["semantic_scaled"], dtype=float)
        recency_scores = np.asarray(score_bundle["recency"], dtype=float)

        def std(x: np.ndarray) -> float:
            return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

        return {
            "semantic_score_mean_all": float(np.mean(semantic_scores)),
            "semantic_score_std_all": std(semantic_scores),
            "semantic_score_min_all": float(np.min(semantic_scores)),
            "semantic_score_max_all": float(np.max(semantic_scores)),
            "recency_score_mean_all": float(np.mean(recency_scores)),
            "recency_score_std_all": std(recency_scores),
            "recency_score_min_all": float(np.min(recency_scores)),
            "recency_score_max_all": float(np.max(recency_scores)),
        }


def retrieve_recency(
    memories: list[dict[str, Any]],
    k: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        memories,
        key=lambda x: x["recency_order"],
        reverse=True,
    )
    return [dict(x) for x in ranked[:k]]


def calculate_recency_scores(
    memories: list[dict[str, Any]],
) -> np.ndarray:
    n = len(memories)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    newest_first = sorted(
        range(n),
        key=lambda i: memories[i]["recency_order"],
        reverse=True,
    )
    scores = np.zeros(n, dtype=float)
    for rank, index in enumerate(newest_first):
        scores[index] = 1.0 - rank / (n - 1)
    return scores


def minmax_normalize(values: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values, False
    low = float(np.min(values))
    high = float(np.max(values))
    if np.isclose(low, high):
        return np.full(values.shape, 0.5, dtype=float), True
    return (values - low) / (high - low), False


def _stable_memory_key(memory: dict[str, Any]) -> tuple:
    return (
        str(memory.get("session_id", "")),
        int(memory.get("round_index", 0)),
        str(memory.get("id", "")),
    )


def stable_rank_indices(
    scores: np.ndarray,
    memories: list[dict[str, Any]],
) -> list[int]:
    """Rank descending with a deterministic memory-identity tie break."""
    return sorted(
        range(len(memories)),
        key=lambda index: (-float(scores[index]), _stable_memory_key(memories[index])),
    )


def stable_ranks(scores: np.ndarray, memories: list[dict[str, Any]]) -> np.ndarray:
    """Return one-based ordinal ranks (no ambiguous tied ranks)."""
    ranks = np.empty(len(memories), dtype=float)
    for rank, index in enumerate(stable_rank_indices(scores, memories), start=1):
        ranks[index] = rank
    return ranks


def retrieve_from_scores(
    memories: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    k: int,
    score_name: str,
    component_scores: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for index in stable_rank_indices(scores, memories)[:k]:
        item = dict(memories[index])
        item[score_name] = float(scores[index])
        for name, values in (component_scores or {}).items():
            item[name] = float(values[index])
        results.append(item)
    return results


def retrieve_semantic(
    query: str,
    memories: list[dict[str, Any]],
    memory_embeddings: np.ndarray,
    retriever: StellaRetriever,
    k: int = 5,
) -> list[dict[str, Any]]:
    cosine_scores = retriever.semantic_cosine_scores(query, memory_embeddings)
    top_indices = np.argsort(cosine_scores)[::-1][:k]

    results = []
    for index in top_indices:
        item = dict(memories[index])
        item["semantic_cosine"] = float(cosine_scores[index])
        results.append(item)
    return results


def retrieve_semantic_from_scores(
    memories: list[dict[str, Any]],
    cosine_scores: np.ndarray,
    k: int = 5,
) -> list[dict[str, Any]]:
    return retrieve_from_scores(
        memories,
        cosine_scores,
        k=k,
        score_name="semantic_cosine",
    )


def retrieve_hybrid(
    query: str,
    memories: list[dict[str, Any]],
    memory_embeddings: np.ndarray,
    retriever: StellaRetriever,
    k: int = 5,
) -> list[dict[str, Any]]:
    cosine_scores = retriever.semantic_cosine_scores(query, memory_embeddings)

    # Keep the Research Plan's pre-registered formula as the PRIMARY analysis.
    # A sensitivity analysis may compare another normalization, but must not
    # replace this main condition after looking at results.
    semantic_scores = (cosine_scores + 1.0) / 2.0
    recency_scores = calculate_recency_scores(memories)
    hybrid_scores = 0.5 * semantic_scores + 0.5 * recency_scores

    top_indices = np.argsort(hybrid_scores)[::-1][:k]
    results = []
    for index in top_indices:
        item = dict(memories[index])
        item["semantic_score"] = float(semantic_scores[index])
        item["recency_score"] = float(recency_scores[index])
        item["hybrid_score"] = float(hybrid_scores[index])
        results.append(item)
    return results


def retrieve_hybrid_from_scores(
    memories: list[dict[str, Any]],
    score_bundle: dict[str, np.ndarray | bool],
    *,
    strategy: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    if strategy not in {"hybrid_raw", "hybrid_minmax", "hybrid_rrf"}:
        raise ValueError(strategy)
    scores = np.asarray(score_bundle[strategy], dtype=float)
    components = {
        "semantic_score": np.asarray(score_bundle["semantic_scaled"], dtype=float),
        "recency_score": np.asarray(score_bundle["recency"], dtype=float),
    }
    return retrieve_from_scores(
        memories,
        scores,
        k=k,
        score_name=f"{strategy}_score",
        component_scores=components,
    )


def calculate_round_recall_at_k(
    retrieved_memories: list[dict[str, Any]],
    relevant_round_ids: list[str],
) -> float | None:
    relevant = set(relevant_round_ids)
    if not relevant:
        return None
    retrieved = {str(memory["id"]) for memory in retrieved_memories}
    return len(retrieved & relevant) / len(relevant)


def calculate_recall_any_at_k(
    retrieved_memories: list[dict[str, Any]],
    relevant_round_ids: list[str],
) -> float | None:
    relevant = set(relevant_round_ids)
    if not relevant:
        return None
    retrieved = {str(memory["id"]) for memory in retrieved_memories}
    return float(bool(retrieved & relevant))


def calculate_session_recall_at_k(
    retrieved_memories: list[dict[str, Any]],
    gold_session_ids: list[str],
) -> float | None:
    relevant = set(str(x) for x in gold_session_ids)
    if not relevant:
        return None
    retrieved_sessions = {
        str(memory["session_id"]) for memory in retrieved_memories
    }
    return len(retrieved_sessions & relevant) / len(relevant)


# Backward-compatible name. The report should call this round-level recall.
def calculate_recall_at_k(
    retrieved_memories: list[dict[str, Any]],
    relevant_ids: list[str],
) -> float | None:
    return calculate_round_recall_at_k(retrieved_memories, relevant_ids)


def _synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(
    retrieval_function: Callable[[], Any],
    warmup: int = 1,
    repeats: int = 5,
) -> dict[str, Any]:
    for _ in range(warmup):
        retrieval_function()

    _synchronize_if_needed()
    runs_ms: list[float] = []

    for _ in range(repeats):
        _synchronize_if_needed()
        start = time.perf_counter()
        retrieval_function()
        _synchronize_if_needed()
        end = time.perf_counter()
        runs_ms.append((end - start) * 1000.0)

    return {
        "runs_ms": runs_ms,
        "median_ms": median(runs_ms),
    }
