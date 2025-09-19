import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

from lib.scoring_fns import hhi

# todo - testing; consider verifying the logit dimensions to these are always correct

def kl_divergence(p: Tensor, q: Tensor) -> float:
    """Compute KL divergence between two logit distributions."""
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return torch.sum(p * torch.log(p / q)).item()


def cosine_similarity(p: Tensor, q: Tensor) -> float:
    """Compute cosine similarity between two logit distributions."""
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    res = torch.dot(p, q) / (torch.norm(p) * torch.norm(q))
    return res.item()


def euclidean_distance(p: Tensor, q: Tensor) -> float:
    """Compute Euclidean distance between two logit distributions.

    Ranges from 0 to 2. Note that it is very similar to JS divergence except that it is noisier

    """
    warnings.warn("Using euclidean distance.")
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    # (for testing)
    # norm = torch.norm(p - q)
    # assert torch.allclose(norm, vector_norm)

    # (test norm as expected)
    # norm2 = torch.sum((p-q) ** 2)
    # final = norm2 ** (1/2)
    # assert torch.allclose(final, vector_norm)

    vector_norm = torch.linalg.vector_norm(p-q)
    return vector_norm.item()

def _kl_div_test(d1, d2):
    # all_vals = d1 * torch.log2(d1/d2)
    all_vals = d1 * torch.log(d1/d2)
    return all_vals.sum()



def _jensen_shannon_divergence_test(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Computes the Jensen-Shannon Divergence between two probability distributions.

    Args:
        p,q unidimensional logits

    Returns:
        torch.Tensor: Scalar tensor representing the JSD between p and q.
    """
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)

    # Compute the midpoint distribution
    m = (p + q)/2

    res = 0.5 * (_kl_div_test(p, m) + _kl_div_test(q, m))
    return res


def jensen_shannon_divergence(
        p: torch.Tensor,
        q: torch.Tensor,
        check_values = False
) -> float:
    """
    Computes the Jensen-Shannon Divergence between two probability distributions.
    - this uses natural log based on our test

    Args:
        p,q unidimensional logits

    Returns:
        torch.Tensor: Scalar tensor representing the JSD between p and q.
    """
    # our reduction for kl_div requires sum, rather than mean
    assert len(p.shape) == len(q.shape) == 1

    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)

    # Compute the midpoint distribution in log-space
    m_log = torch.log(0.5 * (p_log.exp() + q_log.exp()))

    # Compute KL divergences
    # note that we needed reduction sum; this will not work for batches
    kl_pm = F.kl_div(m_log, p_log, log_target=True, reduction='sum')
    kl_qm = F.kl_div(m_log, q_log, log_target=True, reduction='sum')

    # Compute Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)

    if check_values:
        test_val = _jensen_shannon_divergence_test(p, q)
        assert torch.allclose(test_val, jsd, atol=1e-5), f"{test_val.item()} != {jsd.item()}"

    return jsd.item()


def hhi_dist(p: Tensor, q: Tensor) -> float:
    """Compute Euclidean distance between two logit distributions."""
    p = torch.softmax(p, dim=-1)
    q = torch.softmax(q, dim=-1)
    return abs(hhi(p).item() - hhi(q).item())
