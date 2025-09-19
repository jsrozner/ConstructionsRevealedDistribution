from typing import List

import torch
import torch.nn.functional as F


from lib.common.mlm_singleton import get_singleton_scorer

def top_k_preds_for_logits(logits: torch.Tensor, top_k: int) -> List[str]:
    """
    Get the top_k predictions (words as strings) from a logit vector
    """
    mlm_scorer = get_singleton_scorer()
    predicted_ids = torch.topk(logits, top_k).indices
    predicted_words = [
        mlm_scorer.tokenizer.decode([id])
        for id in predicted_ids
    ]
    return predicted_words

def top_k_preds_with_probs(
        logits: torch.Tensor,
        top_k: int,
        model_name="roberta-large"
) -> list[tuple[str, float]]:
    """
    Get the top_k predictions (words as strings) and their probabilities
    from a logit vector.
    """
    mlm_scorer = get_singleton_scorer()
    if mlm_scorer.model.name_or_path != model_name:
        raise Exception("Models do not match")

    # Convert logits to probabilities
    probs: torch.Tensor = F.softmax(logits, dim=-1)

    # Get top-k indices and probabilities
    top_probs, top_ids = torch.topk(probs, top_k)

    # Decode ids to tokens and zip with probabilities
    predicted: list[tuple[str, float]] = [
        (mlm_scorer.tokenizer.decode([idx.item()]), prob.item())
        for idx, prob in zip(top_ids, top_probs)
    ]

    return predicted