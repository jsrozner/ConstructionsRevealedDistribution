from typing import List

import torch

from lib.common.mlm_singleton import init_singleton_scorer

mlm_scorer = init_singleton_scorer('roberta-large', output_attentions=True)

def top_k_preds_for_logits(logits: torch.Tensor, top_k: int) -> List[str]:
    """
    Get the top_k predictions (words as strings) from a logit vector
    """
    predicted_ids = torch.topk(logits, top_k).indices
    predicted_words = [
        mlm_scorer.tokenizer.decode([id])
        for id in predicted_ids
    ]
    return predicted_words
