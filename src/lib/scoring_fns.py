import math
from typing import Callable

import torch

from rozlib.libs.library_ext_utils.utils_torch import round_tensor

ScoreFn = Callable[[torch.Tensor], float]
ScoreFnTensor = Callable[[torch.Tensor], torch.Tensor]

def _gini(x: torch.Tensor) -> torch.Tensor:
    # Ensure the input is a 1D tensor
    assert x.dim() == 1, "Input tensor must be 1-dimensional"

    # Step 1: Sort the tensor values in ascending order
    sorted_x, _ = torch.sort(x)

    # Step 2: Compute the index-weighted sum
    n = x.size(0)
    index = torch.arange(1, n + 1, dtype=torch.float32)  # 1-based indexing
    weighted_sum = torch.sum((2 * index - n - 1) * sorted_x)

    # Step 3: Calculate the Gini coefficient
    # gini = (2.0 * weighted_sum) / (n * torch.sum(sorted_x))
    gini = (2.0 * weighted_sum) / n

    return gini

def _entropy_range_rounded(logits: torch.Tensor, start: int, end: int) -> float:
    # p = exp(log(p))
    # log(p) * p

    logits, logits_idxs = logits.sort(descending=True)
    selected_probs = logits[logits_idxs[start:end]]

    selected_probs = torch.softmax(selected_probs, dim=-1)
    return round_tensor(-torch.sum(selected_probs * torch.log(selected_probs)), 2)

def entropy(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs))
    return entropy

def entropy_rounded(logits: torch.Tensor):
    e = entropy(logits)
    return round_tensor(e, 2)

def hhi(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    hhi = torch.sum(probs*probs)
    return hhi

def hhi_rounded(logits: torch.Tensor):
    h = hhi(logits)
    return round_tensor(h, 3)

def min_surprisal(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    max_prob = torch.max(probs)
    # lowest surprisal
    # return 0
    return -torch.log(max_prob)

def min_surprisal_rounded(logits: torch.Tensor):
    s = min_surprisal(logits)
    return round_tensor(s, 3)

def get_logits(logits: torch.Tensor, tok_id: torch.Tensor) -> torch.Tensor:
    return logits[tok_id]

def surprisal(logits: torch.Tensor, tok_id: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    prob_orig_token = probs[tok_id]
    return -torch.log(prob_orig_token)

def probability(logits: torch.Tensor, tok_id: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    prob_orig_token = probs[tok_id]
    return prob_orig_token

def surprisal_rounded(logits: torch.Tensor, tok_id: torch.Tensor):
    s = surprisal(logits, tok_id)
    return round_tensor(s, 3)


def step_score(logits: torch.Tensor, min_prob=0.01, num_to_consider=10):
    """
    Characterizes how "steppy" the distribution is

    Whenever a step from one to the next is small (i.e. similar prob),
    that is "rewarded"

    It's the sum over prob[i]/next_prob weighted by prob[i]
    """
    assert logits.dim() == 1
    # probs = torch.softmax(logits, dim=-1)

    top_logits, top_ids = torch.topk(logits, num_to_consider)
    probs = torch.exp(top_logits - top_logits[0])
    # one_by_one_diffs = torch.diff(top_logits)
    # one_by_one_diffs = torch.cat((
    #     torch.Tensor([top_logits[0]]),
    #     one_by_one_diffs)
    # )
    # one_by_one_probs = torch.exp(one_by_one_diffs)
    one_by_one_probs = torch.exp(torch.diff(top_logits))

    score = 0.0
    for i in range(1, num_to_consider):
        if not probs[i] > min_prob:
            break
        new_score = 1.0
        new_score *= one_by_one_probs[i-1] # upweight by how "unsteppy" it is
        # penalize low probability, but penalize less the further out it is
        new_score *= 1/math.pow(1/probs[i], 1/i)

        score += new_score

    return score


def _josh_score_rounded(logits: torch.Tensor, num_to_consider: int =10):
    """
    Characterizes how "steppy" the distribution is

    Whenever a logit in the top num_to_consider has a big step (e.g. prob is
    much bigger than next one, score will be high)

    It's the sum over prob[i]/next_prob weighted by prob[i]
    """
    assert logits.dim() == 1
    probs = torch.softmax(logits, dim=-1)

    top_ids = torch.topk(logits, 10).indices
    # top_logits = logits[top_ids]
    top_probs = probs[top_ids]

    # probs = torch.exp(top_logits)

    total = 0
    for i in range(num_to_consider - 1):
        total += (top_probs[i]/top_probs[i+1] - 1) * top_probs[i]
    return round_tensor(total, 2)

def hhi_trunc_rounded(logits: torch.Tensor, k_to_omit: int=2):
    # sort and remove the k_to_omit
    # print(torch.softmax(logits, dim=-1)[:3])
    # print(logits[:3])
    probs, probs_indxs = logits.sort(descending=True)
    probs = probs[k_to_omit:]

    probs = torch.softmax(probs, dim=-1)
    # print(probs[:3])

    ret = round_tensor(torch.sum(probs*probs), 3)
    # print(ret)
    return ret

# todo(low): the return type is wrong
def hhi_ratio_unrounded(logits: torch.Tensor, k_to_omit: int=2):
    hhi_score = hhi_rounded(logits)
    hhi_trunc_score = hhi_trunc_rounded(logits, k_to_omit)
    return hhi_score/hhi_trunc_score


def prob_ratio(token_logits: torch.Tensor) -> torch.Tensor:
    token_probs = torch.exp(token_logits)
    return token_probs/token_probs[0]
