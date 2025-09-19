from __future__ import annotations

from dataclasses import dataclass
from typing import List

from lib.exp_common.config import get_config

"""
Dataclasses for storage of mlm scoring outputs
(originally for exp2)
"""

config = get_config()

@dataclass
class MLMResultForSentence:
    """
    One of these dataclasses per sentence, stored in jsonl format
    """
    file_id: str        # matches the file id, unique
    sentence_id: int    # which sentence in order

    sentence: str       # full sentence

    # note: these are generally just "scores", not HHI scores, which were used only in initial experiments
    # todo: support other scores?
    #  scores: List[Tuple[str, float]]    # score_name: score
    hhi_scores: List[float]

    # mlm_results: List[MLMResultForWord]
    multi_tok_indices: List[int]

    # corresponds to the row of the matrix
    perturbed_sentences: List[str]

    # 2x2 matrix of affinity scores as change in HHI
    # score_matrix_hhi: torch.Tensor | None = None

    # 2x2 matrix of affinity scores as euclidean diff in distribution
    # todo: we should be rounding to reduce storage size
    score_matrix_distribution: List[List[float]]
