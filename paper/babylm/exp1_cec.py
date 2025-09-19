import logging
from collections import defaultdict
from typing import List, Dict

from sklearn.metrics import roc_curve, auc

from affinity.corr_matrix_new import get_affinity_for_word
from corpus_tools.zhou_cxs_so_difficult.corpus_leonie_eap_aap_cec import BaseExample, get_data
from proj.cxs_are_revealed.paper.data_config import Exp1Zhou


# todo: this probably should not go here

def get_affinity_for_so(
        sent: str,
        debug=False,
):
    return get_affinity_for_word(sent, "so", debug)

def get_hists(ex_list: List[BaseExample], debug=False):
    """
    For all the aligned results, accumulate according to the phrase type (4 types)
    the hhi scores
    """
    # we get a histogram of the HHI scores
    hists: Dict[int, List[float]] = defaultdict(list)
    err_ct = 0
    if debug:
        exs_to_run = ex_list[:1]
    else:
        exs_to_run = ex_list
    for ex in exs_to_run:
        sent = ex.sentence_punct_fixed
        if debug: print(sent)
        try:
            so_prob = get_affinity_for_so(sent)
            if debug: print(so_prob)
            hists[ex.label.value].append(so_prob)
        except Exception as e:
            # traceback.print_exc()
            print(e)
            err_ct += 1
    logging.error(f"error count: {err_ct}")

    return hists, err_ct


def merge_oce_cec(hists_probabilities):
    h = [hists_probabilities[x] for x in range(1,5)]

    # merge oce and cec into a single class
    hists_merge_oce_cec = []

    hists_merge_oce_cec.append(h[0])
    hists_merge_oce_cec.append(h[1])
    hists_merge_oce_cec.append(h[2])
    hists_merge_oce_cec[2].extend(h[3])     # cec / oce

    return hists_merge_oce_cec

def calc_acc_score(all_exs, hist) -> float:
    ct = 0
    for v in hist[2]:
        if v <= 0.78:
            ct += 1

    for i in [0, 1]:
        for v in hist[i]:
            if v > 0.78:
                ct += 1
    total = len(all_exs)
    return (total-ct) /total

def get_calc_roc_auc(hist) -> float:
    labels = []
    scores = []
    for v in hist[2]:
        labels.append(1)
        scores.append(v)

    for i in [0, 1]:
        for v in hist[i]:
            labels.append(0)
            scores.append(v)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Step 3: Compute AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# adapted from cec_suite.ipynb
def exp1_cec(**kwargs):
    # get the data
    # note that this will set minimal_clean=True (so it includes examples that were omitted in the original Words in Distrib paper)
    exs_no_errors = get_data(Exp1Zhou.zhou_original_xlsx)
    hists, err_ct = get_hists(exs_no_errors)

    # merge cec, oce
    hists_merged_oce_cec = merge_oce_cec(hists)

    acc_score = calc_acc_score(exs_no_errors, hists_merged_oce_cec)
    roc_auc = get_calc_roc_auc(hists_merged_oce_cec)

    return acc_score, roc_auc, err_ct