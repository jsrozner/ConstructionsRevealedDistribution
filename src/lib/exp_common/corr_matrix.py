import logging
import random
import warnings
from curses.ascii import isalpha, islower, isupper
from pprint import pp
from typing import Tuple, List, Optional, Callable, Literal

import torch

from lib.common.corr_matrix_common import top_k_preds_for_logits
from lib.utils.utils_misc import replace_word_with_substitution
from rozlib.libs.library_ext_utils.utils_transformer import ROBERTA_SPACE_START_CHAR
from rozlib.libs.utils.string import str_all_punct, split_and_remove_punct
# from scoring.exp2_affinity_matrix.dataclass_exp2 import MLMResultForSentence
from lib.distr_diff_fcns import euclidean_distance
from lib.exp_common.sentence_for_mlm_processing import SentenceForMLMProcessing
from lib.exp_common.mlm_align_tokens import reassembled_words_from_tokens_roberta, TokenizedWordInSentence, align_words_with_token_list
from lib.common.mlm_singleton import get_singleton_scorer
from lib.scoring_fns import hhi, surprisal, probability
from lib.exp_common.mlm_result_for_sentence import MLMResultForSentence

# from libs.utils import replace_word_with_substitution
# from libs.utils import split_and_remove_punct, str_all_punct
# from libs.library_ext_utils.utils_transformer import ROBERTA_SPACE_START_CHAR

mlm_scorer = get_singleton_scorer()

"""
Attention processing
"""
def process_attn(attentions: Tuple[torch.Tensor]):
    # Step 4: Stack all layers' attention tensors into a single tensor along a new dimension
    # Shape: (num_layers, num_heads, seq_length, seq_length)
    attention_tensor_all_layers = torch.stack(attentions, dim=0)
    print(attention_tensor_all_layers.shape)
    attention_tensor_one_batch = attention_tensor_all_layers.squeeze(1)
    print(attention_tensor_one_batch.shape)

    # Step 5: Perform max-pooling over layers and heads
    # Shape: (seq_length, seq_length)
    max_pooled_attention = torch.amax(attention_tensor_one_batch, dim=(0, 1))
    print(max_pooled_attention.shape)
    return max_pooled_attention.to('cpu')

# get attentions for producing a 2-d matrix
def get_base_attns(sent):
    attns, tokens = mlm_scorer.get_attns_for_sentence(sent)
    attns_pooled = process_attn(attns)
    return attns_pooled, tokens

"""
Logits handling
"""
def make_logits_list(sent: str) -> Tuple[List[torch.Tensor], List[int]]:
    """
    For a given input sentence, return a Tuple of
    - List of logit tensors (one for each idx in sentence; none if multiply tokenized)
    - List of indices in the sentence that are multiply tokenized

    #todo
    Note that this duplicates the logic in mlm.py - MLMScorer.print_preds_all
    - It probably also duplicates some logic in mlm_gpu
    """
    all_logits: List[torch.Tensor] = []
    multi_token_idx_list = []
    for idx, w, logits in mlm_scorer.get_logits_for_all_words_in_sentence(sent):
        if logits is None:
            multi_token_idx_list.append(idx)
            all_logits.append(torch.zeros(mlm_scorer.tokenizer.vocab_size))
        else:
            all_logits.append(logits)
    return all_logits, multi_token_idx_list

# todo: this mostly duplicates above fcn
def make_logits_list_under_substit(
        sent: str,
        orig_word: str,
        word_idx_to_sub: int,
        substit_word: str
) -> Tuple[str, List[torch.Tensor], List[int]]:
    """
    For a given input sentence, return a Tuple of
    - List of logit tensors (one for each idx in sentence; none if multiply tokenized)
    - List of indices in the sentence that are multiply tokenized

    # todo
    Note that this duplicates the logic in mlm.py - MLMScorer.print_preds_all
    - It probably also duplicates some logic in mlm_gpu
    """
    all_logits: List[torch.Tensor] = []
    multi_token_idx_list = []
    new_sent = replace_word_with_substitution(
        sent, orig_word, word_idx_to_sub, substit_word,
        # do_print=True
    )
    for idx, w, logits in mlm_scorer.get_logits_for_all_words_in_sentence(new_sent):
        if logits is None:
            multi_token_idx_list.append(idx)
            all_logits.append(torch.zeros(mlm_scorer.tokenizer.vocab_size))
        else:
            all_logits.append(logits)
    return new_sent, all_logits, multi_token_idx_list


"""
Score matrix
"""
def _draw_word():
    vocab_len = mlm_scorer.tokenizer.vocab_size
    rand_id = random.randint(0, vocab_len - 1)
    s = mlm_scorer.tokenizer.convert_ids_to_tokens(rand_id)
    if s is None:
        print(s)
    return s

def get_subs_list_random(sent_words_list: List[str]):
    subs = []

    # first word should be capitalized
    while 1:
        # redraw if it's not an alphabetic uppercase
        s = _draw_word()
        if not isalpha(s[0]) or not isupper(s[0]):
            continue
        subs.append(s)
        break

    # fill in rest of subs one at a time
    i = 1
    while i < len(sent_words_list):
        print(i)
        # redraw if it's not an alphabetic lowercase word
        s = _draw_word()
        # start with space, then alpha, then should match case
        # todo: not sure why it can be none
        if s is None or not s[0] == ROBERTA_SPACE_START_CHAR or not isalpha(s[1]) or not islower(s[1]) == islower(sent_words_list[i][0]):
            continue
        subs.append(mlm_scorer.tokenizer.convert_tokens_to_string([s]).strip())
        i += 1

    return subs


# todo: this is duplicated
def get_sub_list(
        sent_word_list: List[str],
        preds: List[List[str]],
) -> List[str]:
    all = []
    for idx, w in enumerate(sent_word_list):
        all.append(get_first_different_substitution(w, preds[idx]))
    return all

def get_first_different_substitution(
        w: str,
        w_list: List[str],
        reverse_subs_order=False
):
    w = w.strip().lower()

    if reverse_subs_order:
        subs_ordered = w_list[::-1]
    else:
        subs_ordered = w_list
    for sub in subs_ordered:
        sub_clean = sub.strip().lower()
        # we want the sub to be different, even if it is shorter (e.g. bike and bikes)
        shorter, longer = sorted([w, sub_clean], key=lambda x: len(x))
        if shorter == longer[:len(shorter)]:
            continue
        # don't allow punctuation only
        if str_all_punct(sub.strip()):
            # print(f"{sub} is all punct")
            continue
        else:
            # print(f"{sub} is not all punct")
            # todo: check that this is always a valid transformation for  reproducing the sentence
            return sub.strip()
    print(f"WARN: no valid sub found; using {w_list[0]}")
    return w_list[0].strip()

def calc_probs_for_mi(
        orig_sent: str,
        sent_word_list: List[str],
        logit_list_under_sub,
        multi_tok_indices,
):
    prob_list = []
    # get probabilities of original word
    s = SentenceForMLMProcessing(
        file_id="0",
        sent_id=0,
        sent = orig_sent,
    )
    assert s.sent_words_list == sent_word_list
    def get_token_for_word_at_idx(i):
        curr_word = s.aligned_word_reps[i]
        # assert curr_word.str_rep_as_string_no_space == w
        tokens = curr_word.tokens
        assert len(tokens) == 1, pp(curr_word)
        t = s.sent_input_ids[0][curr_word.tok_idx_start]
        # assert t.shape == (1), f"Shape is {t.shape}; {t}"
        dec = mlm_scorer.tokenizer.decode(t)
        # pp(curr_word)
        assert len(curr_word.tokens_as_strings) == 1
        assert dec == curr_word.tokens_as_strings[0], f"decoded: [{dec}] != [{curr_word.str_rep_no_special}]"
        return t
    for i, logits in enumerate(logit_list_under_sub):
        if i in multi_tok_indices:
            prob_list.append(0)
            continue
        # print("-")
        probs = torch.softmax(logits, dim=-1)
        # print(logits.shape)
        # print(probs.shape)
        t = get_token_for_word_at_idx(i)
        # print(probs[t].shape)
        # print(f"prob of word is probs[{t}] == {probs[t]}")
        prob_list.append(probs[t].item())

    return prob_list


def get_score_matrix(
        orig_sent: str,
        logits_list_base: List[torch.Tensor],
        preds: List[List[str]],
        multi_tok_indices: List[int],
        topk=5,
        subs_list: Optional[List[str]] = None,
        distr_diff_fn: Callable[[torch.Tensor, torch.Tensor], float] = euclidean_distance,
        compute_probs_for_mi=False,
        do_print = False
):
    """
    Get the 2D score matrix by substituting in words at each word position.

    - orig_sent: the original sentence
    - logits_list_base: List of logits for each position with no perturbation / masking
    - preds: For each word in original sentence, the list of top model fills for that position
    (Used for substitutions if a list of subs is not given)
    - multi_tok_indices: List of any indices (by word) where a word was multiply tokenized
    - topk: how many topk logits / preds to return when doing the substitutions
    - subslist - if given, the word to substitute at each position in the sentence (each corresponds to one row in the output matrix)
        if not given, then get_first_different_substitution will be used to retrive a sub from the preds list
    - distr_diff_fn - the function to use to calculate the distributional diff between base logits (no perturbation)
        and under substitution / mask
    - compute_probs_for_mi - an experimental functionality that does not compute the correct thing right now (todo)
    """
    if not do_print:
        print = lambda x: x

    if compute_probs_for_mi:
        logging.warning(f"Compute probs for MI uses an incorrect method for MI calculation")
    sent_word_list = split_and_remove_punct(orig_sent)

    corr_scores = torch.empty((0, len(sent_word_list)))
    # this will be created, but not filled if compute_probs_for_mi is false (it should be empty if compute_probs is false)
    probs_all = torch.empty((0, len(sent_word_list)))
    new_sents_under_sub: List[str] = []

    # outer loop is which word we are perturbing (row in matrix)
    for idx, w in enumerate(sent_word_list):
        print(f"{idx}: {w}")
        if idx in multi_tok_indices:
            print(f"skipping {w}")
            corr_scores = torch.vstack((corr_scores, torch.zeros(len(sent_word_list))))
            if compute_probs_for_mi:
                probs_all = torch.vstack((probs_all, torch.zeros(len(sent_word_list))))
            continue

        if not subs_list:
            word_for_substitution = get_first_different_substitution(w, preds[idx])
        else:
            print(f"using subs list: {subs_list[idx]}")
            word_for_substitution = subs_list[idx]

        # get logits under sub
        # this gets the logits for each word in the sentence when another word is masked
        # e.g., if we mask w2 in a 3 word sentence, we will have (m, m w3), (w1, m, w3), (w1, m, m) => logits at pos [1, 2, 3]
        # this list is ((2,1,1), (2,2,2), (2,3,3))
        # note that make_logits_list_under_sub will print the updated sentence
        new_sent, logit_list_under_sub, _ = make_logits_list_under_substit(
            orig_sent, w, idx, word_for_substitution)
        new_sents_under_sub.append(new_sent)

        # p: List[List[str]] = [top_k_preds_for_logits(l, topk) for l in
        #                       logit_list_under_sub]
        # pp(p)

        # calc scores (this list corresponds to a row in the matrix)
        score_list = []
        assert len(logits_list_base) == len(logit_list_under_sub) == len(sent_word_list)
        # inner loop is columns
        for orig_l, new_l in zip(logits_list_base, logit_list_under_sub):
            # todo: not sure why we are sending them to device (they should still be on device?)
            dist_diff = distr_diff_fn(orig_l.to(mlm_scorer.device), new_l.to(mlm_scorer.device))

            # if normalize:
            # todo: we wanted to get normalized; but we are not returning the
            # score for the MASK
            #
            #     # we need to obtain the representation for the position
            #     # ie. we need ((2,1,2), (2,2,2), (2,3,2))
            #
            #     # we are going to see how much the perturbed spot actually changed
            #     print(sent_word_list[idx])
            #     perturbed_loc_orig_dist = logits_list_base[idx].to(mlm_scorer.device)
            #     perturbed_loc_perturbed_dist = logit_list_under_sub[idx].to(mlm_scorer.device)
            #     perturbed_dist_diff = distr_diff_fn(perturbed_loc_orig_dist, perturbed_loc_perturbed_dist)
            #     divisor = perturbed_dist_diff
            #     print(f"normalize: {divisor}")
            #     # assert(divisor > 0)
            # if divisor == 0:
            #     if abs(dist_diff) > 1e-6:
            #         print(f"divisor is 0 but dist_diff is {dist_diff}")
            #     score_list.append(0)
            # else:
            #     score_list.append(dist_diff / divisor)
            score_list.append(dist_diff)

        # maybe do probs for MI (todo: not done correctly)
        if compute_probs_for_mi:
            raise Exception("not implemented; recheck")
            # instead of accumulating in this function, we accumulate in probs_list in other fcn and then aggregate
            prob_list = calc_probs_for_mi(orig_sent, sent_word_list, logit_list_under_sub, multi_tok_indices)
            probs_all = torch.vstack((probs_all, torch.Tensor(prob_list)))

        corr_scores = torch.vstack((corr_scores, torch.Tensor(score_list)))
        print("----------")

    return corr_scores, new_sents_under_sub, probs_all

def _compute_surprisal_for_logits(
        orig_sent: str,
        logits_list_base: List[torch.Tensor],
        score_fn: Callable
) -> List[torch.Tensor]:
    """
    This is a helper function to handle surprisal computation, since
    surprisal requires knowing the original token
    """
    assert score_fn in [surprisal, probability]
    # todo: we are duplicating code from get_logits_for_word_in_sent_by_idx
    input_ids, tokens = mlm_scorer._prepare_inputs_for_sentence(orig_sent)
    tokenized_words = reassembled_words_from_tokens_roberta(tokens)

    # align words with tokenization
    sent_words_list = split_and_remove_punct(orig_sent)
    aligned: List[TokenizedWordInSentence] = align_words_with_token_list(sent_words_list, tokenized_words)

    scores: List[torch.Tensor] = []
    # use the index to get the original token
    for word_idx, logits in enumerate(logits_list_base):
        aligned_token_word = aligned[word_idx]

        # note that we previously checked for multitokenized
        token_idx: int = aligned_token_word.tok_idx_start

        masked: int = input_ids[0, token_idx]
        scores.append(score_fn(logits, masked))
    return scores

def get_scores(
        orig_sent: str,
        subs_list: Optional[List[str]],
        subs_method: Optional[Literal["related", "random", "mask"]] = None,
        score_fn = hhi,
        num_preds = 5,
        calculate_affinities=True,
        dist_diff_fn=euclidean_distance
) -> Tuple:
    """

    Args:
        orig_sent:
        subs_list: the subs to use; if not given (None), then use subs_method
        subs_method: either "related" or "random" - see functions defined above
        score_fn:
        num_preds:
        calculate_affinities:
        dist_diff_fn:

    Returns:
        A tuple of matrix_scores, new_sents, multi_tok_indices, sent_word_list, scores_floats, preds, probs, subs_list
    """
    logits_list_base, multi_tok_indices = make_logits_list(orig_sent)

    # surprisal requires two arguments so we gotta hack
    if score_fn == surprisal or score_fn == probability:
        scores = _compute_surprisal_for_logits(orig_sent, logits_list_base, score_fn=score_fn)
    else:
        warnings.warn(f"Using non-surprisal / prob score fn: {score_fn}")
        scores: List[torch.Tensor] = list(map(score_fn, logits_list_base))

    scores_floats = list(map(lambda x: x.item(), scores))
    preds: List[List[str]] = [top_k_preds_for_logits(l, num_preds) for l in
                              logits_list_base]

    sent_word_list = split_and_remove_punct(orig_sent)

    if calculate_affinities:
        if subs_list is None:
            assert subs_method is not None
            if subs_method == "mask":
                subs_list = ["<mask>"] * len(orig_sent.split())
            else:
                warnings.warn(f"Using non-mask subs method {subs_method}")
                if subs_method == "related":
                    subs_list = get_sub_list(sent_word_list, preds)
                elif subs_method == "random":
                    subs_list = get_subs_list_random(sent_word_list)
                else:
                    raise Exception("Invalid subs_method give")

        # todo: don't recompute for multiple score functions
        matrix_scores, new_sents, probs = get_score_matrix(
            orig_sent,
            logits_list_base,
            preds,
            multi_tok_indices,
            subs_list=subs_list,
            distr_diff_fn=dist_diff_fn
        )
    else:
        matrix_scores = torch.zeros((1, len(sent_word_list)))   # wrong size, intentionally
        new_sents = [orig_sent]     # wrong size, intentionally
        probs = torch.zeros((1, len(sent_word_list)))   # probably wrong size, intentionally
    return (
        matrix_scores,
        new_sents,
        multi_tok_indices,
        sent_word_list,
        scores_floats,
        preds,
        probs,
        subs_list
    )

def get_info_for_mlm_sentence(
        mlm_result: MLMResultForSentence,
        subs_list: List[str]
):
    """
    This is used to process exp2 affinity GPU results. It duplicates some of the
    above code so that we can post-process the results
    (todo: don't duplicate)
    """
    sent_word_list = split_and_remove_punct(mlm_result.sentence)
    assert len(sent_word_list) == len(subs_list), f"\t{sent_word_list} len != \n\t {subs_list}"

    new_sents_under_sub: List[str] = []

    for idx, w in enumerate(sent_word_list):
        # print(f"{idx}: {w}")
        if idx in mlm_result.multi_tok_indices:
            # print(f"skipping {w}")
            # corr_scores = torch.vstack((corr_scores, torch.zeros(len(sent_word_list))))
            continue

        new_sent = replace_word_with_substitution(
            mlm_result.sentence,
            w,
            idx,
            subs_list[idx],
            do_print=False
        )
        new_sents_under_sub.append(new_sent)

    return sent_word_list, new_sents_under_sub
