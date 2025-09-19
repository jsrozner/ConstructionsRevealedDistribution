import statistics
from typing import Tuple, List

import torch

from affinity.corr_matrix_new import get_logits_for_masked_sent
from affinity.tokenization import Sentence, MultiTokenException
from babylm.exp4_cogs import CogEntryTokenized, get_tokenized_cogs_data
from lib.common.mlm_singleton import get_singleton_scorer
from proj.cxs_are_revealed.paper.cxns_in_distrib.exp5_cc.exp9_cc_utils import is_comparative_with_context, trim_punctuation


# note this was copied from exp9_compcorr in originaly paper
def score_distrib(
        orig_sent: str,
        offset: Tuple[int, int],
        logits: torch.Tensor,
        top_p=0.98,
        top_k: int | None = 100
) -> Tuple[float, List[int]]:
    mlm_scorer = get_singleton_scorer()

    # get sorted
    logit_indices_sorted = torch.argsort(logits, descending=True)
    probs = torch.softmax(logits, dim=-1)
    total_prob_mass = 0
    comparative_prob_mass = 0
    idx = 0
    next_pct = 0.1
    percentiles: List[int] = []
    while total_prob_mass < top_p:
        if top_k and idx > top_k:
            print(f"idx is at {idx} for {orig_sent}, p is {total_prob_mass}")
            break
        # loop iteration
        orig_logit_id = logit_indices_sorted[idx]
        idx += 1    # increment for next
        p = probs[orig_logit_id].item()
        total_prob_mass += p

        # keep track of count fills needed to get to given percentile
        while total_prob_mass > next_pct:
            next_pct += 0.1
            percentiles.append(idx + 1)

        # also accumulate in total if it's comparative
        word = mlm_scorer.tokenizer.decode([orig_logit_id]).strip(" ") # generally has a space

        # note this is the tokenized sent (i.e. extra spaces)
        sent_with_word = (
                orig_sent[:offset[0]] +
                word +
                orig_sent[offset[1]:])
        if is_comparative_with_context(sent_with_word, (offset[0], offset[0] + len(word))):
            comparative_prob_mass += p

    print(percentiles)
    return comparative_prob_mass/ total_prob_mass, percentiles


def get_all_scores(
        comp_corrs: List[CogEntryTokenized],
        use_tf: bool,
        top_k: int | None = None,
        top_p = 0.98,
) -> Tuple[List[float], List[List[int]], int, int]:
    ok = 0
    ct_multi = 0
    err = 0
    all_scores = []
    all_percentiles: List[List[int]] = []
    for cc in comp_corrs:
        words = cc.sent_orig.split(" ")
        # note that the_idx is the index in words, does not count punctuation as words
        for offset, the_idx in zip(cc.tgt_word_offsets, cc.tgt_words):
            assert words[the_idx].lower() == "the"
            comp_adv_adj = words[the_idx + 1]
            comp_word_clean = trim_punctuation(comp_adv_adj)
            offset_start = cc.sent.find(comp_word_clean, offset[1])
            offset_end = offset_start + len(comp_word_clean)

            # todo: we probably want to count comp_word_clean + " "
            # otherwise we migth find an occ that is pluralized, inflected etc
            occ_ct = cc.sent[:offset_end].count(comp_word_clean)

            if not is_comparative_with_context(
                    cc.sent,
                    (offset_start, offset_end),
                    use_tf = use_tf
            ):
                print(f"In {cc.sent}, {comp_word_clean} not comp")
            else:
                ok+=1

            # test that we can tokenize it for mlm
            try:
                sent = Sentence(cc.sent, allow_non_alignment_in_tokenization=True)
            except:
                err += 1
                continue

            # if sent.
            if sent.word_encodings is None:
                err += 1
                continue

            # will error if multitoken or if multiple occurrences of the word
            # fix 1 -> 0 index in occ
            try:
                masked_sent = sent.get_input_with_word_masked(comp_word_clean, occ=occ_ct -1)
            except MultiTokenException as e:
                ct_multi += 1
                continue
            logits = get_logits_for_masked_sent(masked_sent)
            score, percentiles = score_distrib(
                cc.sent,
                (offset_start, offset_end),
                logits,
                top_p=top_p,
                top_k=top_k
            )
            all_scores.append(score)
            all_percentiles.append(percentiles)

    print(f"processed {ok} OK")
    print(f"errors {err}")
    return all_scores, all_percentiles, ct_multi, err


def exp5_cc(**kwargs):
    # get the data
    cogs_data = get_tokenized_cogs_data()
    comp_corrs = [c for c in cogs_data if c.cx_type == 'Comparative Correlative' ]

    # todo: pass as params use_tf and top_k
    all_scores, all_percentiles, multi_err, err = get_all_scores(
        comp_corrs,
        use_tf=False,
        top_k=None,
        top_p=0.85
    )

    score = statistics.mean(all_scores)

    # avg percentiles
    avg_percentiles = [
        statistics.mean([x for x in p_list if x is not None])
        for p_list in zip(*all_percentiles)
        # for p_list in zip_longest(*all_percentiles)
    ]
    print(score, avg_percentiles, multi_err, err)

    return (
        score, multi_err, err, avg_percentiles
    )
