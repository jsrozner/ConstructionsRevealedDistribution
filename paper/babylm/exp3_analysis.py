import logging
import os
import sys

from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

from affinity.tokenization import Sentence, WordEncoding
import numpy as np
from collections import Counter
from typing import List, NamedTuple, Tuple, Optional
from pprint import pp
from typing import Dict

from babylm.stringdb import StringSetDB
from cxns_in_distrib.exp3_magpie.corpus_magpie import MAGPIE_entry, MLMResultForSentenceExp6, get_all_magpie_json
from cxns_in_distrib.exp3_magpie.magpie_processing_helpers import get_all_magpie_results_unclean, check_data2
from proj.cxs_are_revealed.paper.data_config import BabyLMExp3Magpie
from lib.common.mlm_singleton import init_singleton_scorer
from proj.cxs_are_revealed.paper.babylm.config_babylm import BabyLMModelList

stringdb: Optional[StringSetDB] = None

def get_mag_id(id: int, idx: Optional[int] = None):
    if idx:
        return f"{id}::{idx}"
    return f"{id}::"


class MagpieWithSentence(NamedTuple):
    magpie_entry: MAGPIE_entry
    sentence: Sentence

def get_magpie_pretty(magpie_json: List[dict]) -> List[MagpieWithSentence]:
    magpie_entries = [MAGPIE_entry.from_dict(j) for j in magpie_json]
    ret_list = []
    # idxs_to_remove = []
    for i, me in enumerate(magpie_entries):
        s = Sentence(me.sent, allow_non_alignment_in_tokenization=True)
        if s.word_encodings is None:
            print(i)
            # remove from the json
            # idxs_to_remove.append(i)
            # continue

        ret_list.append(MagpieWithSentence(
            me,
            s
        ))
    # if len(idxs_to_remove) > 0:
    #     print(f"will remove {len(idxs_to_remove)} from json")
    #     [magpie_json.pop(i) for i in idxs_to_remove]
    return ret_list

def idiom_word_filter(
        word_encoding: WordEncoding,
        log_counter: Counter,
        min_chars_per_word = None,
) -> bool:
    """
    Return True if the WrappedIdiomWord should be kept
    """
    if len(word_encoding.token_ids) != 1:
        log_counter['filtered_word_multitok'] += 1
        return False

    if min_chars_per_word and len(word_encoding.word_chars) < min_chars_per_word:
        log_counter['filtered_word_too_short'] += 1
        return False

    return True

def get_word_encoding_for_offset(sent: Sentence, offset: Tuple[int,int]) -> Optional[WordEncoding]:
    for we in sent.word_encodings:
        if we.offset == offset:
            return we
    # print(f"for [{sent.sent}], offseted word [{sent.sent[offset[0]:offset[1]]}] not found")
    return None

def get_word_level_scores_with_filtering(
        magpie_list: List[MagpieWithSentence],
        result_map: Dict[int, MLMResultForSentenceExp6],
        min_chars_per_word = None,
        min_sent_length = None
        # print_interesting_examples = False
):
    assert stringdb

    log_counter = Counter()
    # todo: we did not actually set all of them..
    for x in ["err_encoding", "err_no_match_offset", "err_multitok", "filtered_word_too_short",
              "err_unknown", "error"]:
        # todo (note): set to 1 to preserve a value
        log_counter[x] = 1
    all_scores_idiom: List[float] = []
    all_scores_literal: List[float] = []
    for mag_with_sent in tqdm(magpie_list, file=sys.stdout):
        mag_entry = mag_with_sent.magpie_entry
        mag_entry_id_for_string_db = get_mag_id(mag_entry.id)

        # top level whole entry excluded
        if mag_entry_id_for_string_db in stringdb:
            log_counter['common_skip_whole_sent'] += len(mag_entry.offsets)
            continue

        sent = mag_with_sent.sentence
        if sent.word_encodings is None:
            # this happens very rarely
            stringdb.add(mag_entry_id_for_string_db)
            log_counter["err_encoding"] += len(mag_entry.offsets)
            continue

        # these were originally implemneted in magpie_processing_helpers
        result = result_map.get(mag_entry.id)
        if not result or result.did_error:
            # todo: we changed this
            log_counter['error'] += len(mag_entry.offsets)
            stringdb.add(mag_entry_id_for_string_db)
            continue
        if mag_entry.confidence < 0.99:
            log_counter['filtered_confidence'] += len(mag_entry.offsets)
            stringdb.add(mag_entry_id_for_string_db)
            continue
        if min_sent_length and len(mag_entry.sent.split(" ")) < min_sent_length:
            log_counter['filtered_short_sent'] += len(mag_entry.offsets)
            stringdb.add(mag_entry_id_for_string_db)
            continue
        # prev we also checked minicons score

        assert result.sentence == mag_entry.sent, f"{mag_entry.id}"
        idiom_word_encodings = [get_word_encoding_for_offset(sent, x) for x in mag_entry.offsets]

        for idx, iwe in enumerate(idiom_word_encodings):
            # NOTE: this is word-specific filtering, not sentence level
            mag_id = get_mag_id(mag_with_sent.magpie_entry.id, idx)
            # check common table
            if mag_id in stringdb:
                log_counter['common_skip_word'] += 1
                continue

            if iwe is None:
                log_counter["err_no_match_offset"] += 1
                stringdb.add(mag_id)
                continue

            if len(iwe.token_ids) != 1:
                stringdb.add(mag_id)
                log_counter['err_multitok'] += 1
                continue

            if min_chars_per_word and len(iwe.word_chars) < min_chars_per_word:
                stringdb.add(mag_id)
                log_counter['filtered_word_too_short'] += 1
                continue

            try:
                scores = result.scores[1]
                score = scores[iwe.word_idx_in_sent]
            except:
                log_counter["err_unknown"] += 1
                stringdb.add(mag_id)
                continue

            if mag_entry.label == 'i':
                all_scores_idiom.append(score)
            elif mag_entry.label == 'l':
                all_scores_literal.append(score)
            else:
                stringdb.add(mag_id)
                # log_counter["err_invalid_label"] += 1
                raise Exception("invalid label")

            # if print_interesting_examples and score_to_use == 'hhi':
            #     # print particular failure cases - either high hhi and literal or low hhi and idiomatic
            #     if score > 0.95 and mag_wrapper.magpie_entry.label == 'l':
            #     # if score < 0.05 and mag_wrapper.magpie_entry.label == 'i':
            #         print(f"{mag_wrapper.magpie_entry.sent}\n\t "
            #               f"idiom: {mag_wrapper.magpie_entry.idiom}\n\t"
            #               f"tok: {t.idiom_word_chars}")
    pp(log_counter)
    return all_scores_idiom, all_scores_literal, log_counter

def get_roc_score(idioms_scores, literals_scores):
    scores = np.array(idioms_scores + literals_scores)
    # roc curve wants lower values to be positive examples
    # idioms have higher values, so they are negative examples (0s)
    labels = np.array([1] * len(idioms_scores) + [0] * len(literals_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Step 3: Compute AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)
    return roc_auc

# notes for paper results: can vary:
# - whether stringdb is used
# - whether filter is set for min word length and min sent length

def exp3_magpie_process(model_idx: int, test_idx: int) -> Tuple:
    global stringdb
    # todo: indicate with a flag whether db should be updated
    stringdb = StringSetDB("./ssdb.db")
    # stringdb = StringSetDBNoOp("./ssdb.db")

    # get model from cmd line params so we can access data and init the correct mlm
    model_long = list(BabyLMModelList.model_list.keys())[model_idx]
    model_short = BabyLMModelList.model_list[model_long]
    print(model_long, model_short)

    # make sure we have the correct model initialized; no-op
    _ = init_singleton_scorer(model_long)

    data_dir = BabyLMExp3Magpie._exp3_magpie_cluster_out / model_short / "magpie_unclean"
    logging.info(os.getcwd())
    logging.info(os.path.abspath(data_dir))
    if not os.path.exists(data_dir):
        logging.error(f"{data_dir} does not exist)")
        return None,

    # extra data
    return_data = Counter()

    # read in results
    all_magpie_results = get_all_magpie_results_unclean(
        data_dir= data_dir
    )
    result_map: Dict[int, MLMResultForSentenceExp6] = {res.sentence_id : res
                                                       for res in all_magpie_results}
    # print(f"total results: {len(all_magpie_results)}")
    return_data["results"] = len(all_magpie_results)

    # read in the output json
    all_magpie_json = get_all_magpie_json()
    all_magpie_with_sent = get_magpie_pretty(all_magpie_json)
    # print(len(all_magpie_json))

    # print(f"magpie json length (should match): {len(all_magpie_json)}")
    # print("num idioms", len([m for m in all_magpie_with_sent
    #                          if m.magpie_entry.label == 'i']))

    # verify data alignment
    fail_ct = check_data2(all_magpie_with_sent, result_map)
    return_data["err_check_failed"] = max(fail_ct, 1)

    # todo: remove - moved this logic into function below
    # filter
    # filtered_magpie, log_counter = filter_magpie(
    #     all_magpie_with_sent,
    #     result_map,
    #     # all_magpie_minicons_scores,
    #     # min_sent_length=10,   #todo
    # )
    # return_data += log_counter

    # compute scores
    id_scores, lit_scores, log_counter = get_word_level_scores_with_filtering(
        all_magpie_with_sent,
        result_map,
        # print_interesting_examples=True

        # if you want to filter (additional results)
        min_sent_length=10,
        min_chars_per_word=4,
    )
    return_data += log_counter

    roc = get_roc_score(id_scores, lit_scores)
    pp(return_data)
    return (
        roc,
        *tuple(return_data.values())
    )


