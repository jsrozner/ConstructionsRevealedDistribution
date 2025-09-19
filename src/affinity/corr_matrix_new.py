# todo: rename this file
import warnings
from itertools import permutations
from pprint import pp
from typing import Tuple, List, Optional, Callable, Literal, Any

import torch

from affinity.tokenization import Sentence, MaskedSent
from lib.common.corr_matrix_common import top_k_preds_for_logits, top_k_preds_with_probs
from lib.distr_diff_fcns import euclidean_distance, jensen_shannon_divergence
from lib.common.mlm_singleton import get_singleton_scorer
from lib.scoring_fns import surprisal, probability, get_logits


def get_logits_for_each_word_in_sent(
        s: Sentence,
) -> List[Tuple[MaskedSent, bool, torch.Tensor]]:
    """
    Process sentence; will skip multitoken masks

    Args:
        s: Sentence

    Returns: Tuple:
        - MaskedSent
        - bool: isMultiToken
        - logit tensor (if not multitoken)
    """
    mlm = get_singleton_scorer()
    tuples: List[Tuple[MaskedSent, bool, torch.Tensor]] = []
    ct = 0  # todo(low): return this from the generator instead
    for masked_sent in s.inputs_for_each_word(allow_multi_token=True):
        word = s.words_clean[ct]
        if len(masked_sent.masked_token_indices) > 1:
            logits = torch.zeros(mlm.tokenizer.vocab_size)
            tuples.append((masked_sent, True, logits))
            warnings.warn(f"multitoken will be skipped: {word}")
        else:
            logits = get_logits_for_masked_sent(masked_sent)
            tuples.append((masked_sent, False, logits))
        ct += 1

    return tuples

def get_logits_for_masked_sent(
        masked_sent: MaskedSent,
        do_print = False
):
    """
    Given a masked sentence, run forward and get logits at the masked index.
    Note that MaskedSent represents the location that was originally masked.

    This is just a helper function to run forward and extract a single output logits vector

    Args:
        masked_sent:

    Returns: output logits for masked_sent.input_ids run forward; with the logits at the masked index extracted

    """
    mlm = get_singleton_scorer()
    if not do_print:
        p = lambda *args: None
    else:
        p = print

    p("getting logits_for_masked_sent")
    assert len(masked_sent.masked_token_indices) == 1
    outputs = mlm.get_model_outputs_for_input(masked_sent.input_ids)
    logits = outputs.logits

    p("input Ids", masked_sent.input_ids)
    p("tokens", mlm.tokenizer.convert_ids_to_tokens(masked_sent.input_ids[0]))
    p("shape", logits.shape)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    p("outputs - top for each pos")
    top_token_ids = torch.argmax(probs, dim=-1)  # Shape: (words_in_sent,)
    p("top ids", top_token_ids)
    p("tokens", mlm.tokenizer.convert_ids_to_tokens(top_token_ids[0]))

    # predictions is vocab_len [logit, logit, ... logit]
    # the ltg models drop their first <s> token
    # token_logits = logits[0, masked_sent.masked_token_indices[0] + mlm.output_shift_for_decoding]     # batch, idx, then vocab_len shape
    token_logits = mlm.extract_from_tensor_for_batch_and_tok_idx(logits, 0, masked_sent.masked_token_indices[0])

    return token_logits

def get_contextual_repr_for_masked_sent(
        masked_sent: MaskedSent,
        layer = -1,
        do_print = False
):
    """
    Given a masked sentence, run forward and get hidden state at the layer.
    Note that MaskedSent represents the location that was originally masked.

    This is just a helper function to run forward and extract a single output hidden repr

    Args:
        masked_sent:

    Returns: output logits for masked_sent.input_ids run forward; with the logits at the masked index extracted

    """
    mlm = get_singleton_scorer()
    if not do_print:
        p = lambda *args: None
    else:
        p = print

    p("getting logits_for_masked_sent")
    assert len(masked_sent.masked_token_indices) == 1
    outputs = mlm.get_model_outputs_for_input(
        masked_sent.input_ids,
        use_cache=False,
        output_hidden_states=True
    )
    hiddens = outputs.hidden_states
    if hiddens is None:
        raise Exception("hidden states is none")
    single_layer = hiddens[layer]

    # will shift for certain models
    tgt_vector = mlm.extract_from_tensor_for_batch_and_tok_idx(
        single_layer, 0, masked_sent.masked_token_indices[0])

    return tgt_vector

def compute_surprisal_for_logits(
        masked_sent: MaskedSent,
        logits: torch.Tensor,
        # todo: this can actually be only one of two values
        score_fn: Callable[..., torch.Tensor]
) -> float:
    """
    Compute score_fn (either surprisal or probability -- these are just log/exp of each other)
    for "logits[masked_sent.masked_idx]"

    Args:
        masked_sent: Will be used to extract the particular logits at a masked position
        logits: The logits on which to run
        score_fn: Either probability or surprisal

    Returns: the score for the appropriate logit index as float
    """
    if score_fn not in [probability, surprisal, get_logits]:
        raise Exception(f"Invalid score_fn, {score_fn.__name__} passed")
    if len(masked_sent.masked_token_indices) > 1:
        print(f"Warning: multitoken")
        return 0
    orig_token_id = masked_sent.original_tokens_ids[0]

    # compute surprisal
    score = score_fn(logits, orig_token_id)
    return score.item()

def get_scores_new(
        orig_sent: Sentence,
        score_fn = probability,
        num_preds = 5,
        calculate_affinities=True,
        dist_diff_fn=jensen_shannon_divergence,
        normalize = False
) -> Tuple:
    """
    Compute all affinities for each word in orig_sent

    Args:
        orig_sent:
        score_fn:
        num_preds:
        calculate_affinities:
        dist_diff_fn:

    Returns:
        A tuple of
            - local affinities,
            - new sentences (i.e. with <mask> marked),
            - any word indices that were multitokenized
            - global_affinities,
            - num_preds predictions for each word in orig_sent
    """
    if dist_diff_fn != jensen_shannon_divergence:
        warnings.warn("Using non JSD divergence fcn")
    if score_fn != surprisal and score_fn != probability:
        warnings.warn(f"Using non-surprisal / prob score fn: {score_fn}")

    masked_sents, multi_tok_bools, logits_list_base = map(
        lambda x: list(x),
        zip(*get_logits_for_each_word_in_sent(orig_sent))
    )
    # multitok bools is like [true, false, true] => [0,2]; ie get the indices of the trues
    multi_tok_indices = [i for i,x in enumerate(multi_tok_bools) if x]

    global_aff_floats: List[float] = [compute_surprisal_for_logits(ms, l, score_fn)
                                      for ms, l in zip(masked_sents, logits_list_base)]

    preds: List[List[str]] = [top_k_preds_for_logits(l, num_preds)
                              for l in logits_list_base]

    sent_word_list = orig_sent.words_clean

    if calculate_affinities:
        # recomputation is okay because we're using a cache
        local_affinities, new_sents = get_score_matrix_new(
            orig_sent,
            logits_list_base,
            multi_tok_indices,
            distr_diff_fn=dist_diff_fn,
            normalize=normalize
        )
    else:
        # populate these with wrong size
        local_affinities = torch.zeros((1, len(sent_word_list)))   # wrong size, intentionally
        new_sents = [orig_sent.sent]     # wrong size, intentionally
        # probs = torch.zeros((1, len(sent_word_list)))   # probably wrong size, intentionally
    return (
        local_affinities,
        new_sents,
        multi_tok_indices,
        global_aff_floats,
        preds,
    )

def compute_single_local_affinity(
        sent: Sentence,
        index_to_check: int,
        index_to_perturb: int,
        expect_word_at_index_to_check: Optional[str] = None,
        expect_word_at_index_to_perturb: Optional[str] = None,
        distr_diff_fn: Callable[[torch.Tensor, torch.Tensor], float] = jensen_shannon_divergence,
        print_fills_topk: Optional[int] = None,
) -> float:
    """
    (Adapted from get_score_matrix_new)
    Will measure the local_aff bw index_to_check and index_to_perturb.
    In the paper these two correspond to (j,i), since we perturb *rows* and measure *columns*
    Args:
        sent:
        index_to_check:
        index_to_perturb:
        distr_diff_fn:

    Returns:

    """
    mlm = get_singleton_scorer()
    # print(f"computing local aff for {sent.sent}, indices ({index_to_check}, {index_to_perturb})")
    if expect_word_at_index_to_check:
        assert sent.words_clean[index_to_check] == expect_word_at_index_to_check
    if expect_word_at_index_to_perturb:
        assert sent.words_clean[index_to_perturb] == expect_word_at_index_to_perturb

    # todo: assert single mask
    warnings.warn("This function will not assert single masking. You need to check.")

    # compute base distribution
    masked_sent = sent.get_inputs_with_word_idx_masked(index_to_check)
    logits_base = get_logits_for_masked_sent(masked_sent)

    # compute multimask
    # todo: currently this function won't check that the strs match
    warnings.warn(f"This function does not check that the strings match.")
    multi_mask_sent = sent.get_inputs_with_word_indices_masked_multi_mask([index_to_check, index_to_perturb])
    logits_multi_mask = mlm.get_model_outputs_for_input(multi_mask_sent.input_ids).logits
    logits_under_perturb = mlm.extract_from_tensor_for_batch_and_tok_idx(
        logits_multi_mask,
        0,
        multi_mask_sent.masked_words[0].masked_token_indices[0])

    if print_fills_topk:
        preds1 = top_k_preds_with_probs(logits_base, print_fills_topk)
        preds2 = top_k_preds_with_probs(logits_under_perturb, print_fills_topk)
        pp(preds1)
        pp(preds2)

    dist_diff = distr_diff_fn(logits_base, logits_under_perturb)
    dist_diff2 = distr_diff_fn(logits_under_perturb, logits_base)
    assert abs(dist_diff-dist_diff2)/dist_diff < 1e-4
    return dist_diff



def get_score_matrix_new(
        sent: Sentence,
        logits_list_base: List[torch.Tensor],
        multi_tok_indices: List[int],
        distr_diff_fn: Callable[[torch.Tensor, torch.Tensor], float] = euclidean_distance,
        do_print = True,
        normalize = False
):
    """
    Get the 2D score matrix by substituting in words at each word position.

    - orig_sent: the original sentence
    - logits_list_base: List of logits for each position with no perturbation / masking (single masking)
    - multi_tok_indices: List of any indices (by word) where a word was multiply tokenized
    - distr_diff_fn - the function to use to calculate the distributional diff between base logits (no perturbation)
        and under substitution / mask
    """
    mlm = get_singleton_scorer()
    if do_print:
        p = print
    else:
        p = lambda x: None
    if normalize:
        warnings.warn("Normalize does not function as expected; You probably don't want this.")
    sent_word_list = sent.words_clean
    n = len(sent_word_list)
    new_sents_under_sub: List[str] = []

    aff_scores = torch.zeros((n,n))

    # i is row (perturbed); j is column (measured)
    for i,j in permutations(range(len(sent_word_list)), 2):
        # p(f"({i}, {j}): ({sent_word_list[i]}, {sent_word_list[j]}")
        if i in multi_tok_indices or j in multi_tok_indices:
            p(f"skipping multitok")
            continue

        # word_for_substitution = subs_list[idx]
        # this is just so that we can print the sentence! not used for actual masking
        word_for_substitution = "<mask>"
        subbed_sent = sent.get_sent_with_substituted_word_at_idx(i, word_for_substitution)
        new_sents_under_sub.append(subbed_sent)

        multi_mask_sent = sent.get_inputs_with_word_indices_masked_multi_mask([i,j])
        # note that we're still singly tokenized, so masked_token_indices has len 1
        for masked_word in multi_mask_sent.masked_words:
            assert len(masked_word.masked_token_indices) == 1
        outputs = mlm.get_model_outputs_for_input(multi_mask_sent.input_ids)
        logits = outputs.logits
        # i,j are first and second based on call order
        i_logits = mlm.extract_from_tensor_for_batch_and_tok_idx(logits, 0,
                                                                 multi_mask_sent.masked_words[0].masked_token_indices[0])
        j_logits = mlm.extract_from_tensor_for_batch_and_tok_idx(logits, 0,
                                                                 multi_mask_sent.masked_words[1].masked_token_indices[0])

        # dist_diff = distr_diff_fn(orig_l.to(mlm.device), new_l.to(mlm.device)); not sure why we needed to send to device; should have still been on device
        # note that the functions we use are symmetric
        dist_diff = distr_diff_fn(logits_list_base[j], j_logits)
        if normalize:
            dist_diff_at_perturbed_loc = distr_diff_fn(logits_list_base[i], i_logits)
            p(round(dist_diff_at_perturbed_loc, 2))
            p(round(dist_diff, 2))
            score = dist_diff / dist_diff_at_perturbed_loc
        else:
            score = dist_diff
        aff_scores[i,j] = score

    return aff_scores, new_sents_under_sub


# todo: verify this shoudl go here
def get_affinity_for_word(
        sent: str,
        tgt_word: str,
        debug: bool =False,
        score_fn = probability,
        occ: Optional[int] = None   # pass occ around
):
    s = Sentence(sent, allow_non_alignment_in_tokenization=True)    # Sentence class takes care of tokenization

    if debug:
        print(s.encoding)

    # we excluded any sentences that had multiple so's - see the code that reads in clean examples
    masked_sent = s.get_input_with_word_masked(tgt_word, occ=occ)   # pass occ around

    # obtain the logits at the masked position
    logits = get_logits_for_masked_sent(masked_sent, do_print=debug)

    # compute probability of the original word
    prob = compute_surprisal_for_logits(
        masked_sent,
        logits,
        score_fn
    )
    return prob
