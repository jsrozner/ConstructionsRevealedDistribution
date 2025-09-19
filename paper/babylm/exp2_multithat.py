# adapted from cec_suite.ipynb
from typing import List

from affinity.corr_matrix_new import compute_single_local_affinity
from affinity.tokenization import Sentence
from corpus_tools.zhou_cxs_so_difficult.corpus_leonie_eap_aap_cec import BaseExample
from proj.cxs_are_revealed.paper.data_config import Exp1Zhou
from rozlib.libs.common.data.utils_dataclass import read_csv_to_dataclass

def get_data():
    multi_that_exs: List[BaseExample] = read_csv_to_dataclass(BaseExample, Exp1Zhou.cec_multithat_rozner)
    # fix each in place; don't care about result; they are still in multi_that_ex
    _ = [e._fix_self_for_csv_read() for e in multi_that_exs]

    # check no errors
    for m in multi_that_exs:
        assert not m.has_error
    return multi_that_exs

def get_multithat_result(ex: BaseExample):
    s = Sentence(ex.sentence_punct_fixed, allow_non_alignment_in_tokenization=True)
    correct_that_score = compute_single_local_affinity(s, ex.so_idx, ex.that_idx, "so", "that")

    all_that_idxs = [i for i,w in enumerate(s.words_clean) if w.lower() == "that"]
    assert ex.that_idx in all_that_idxs
    assert len(all_that_idxs) > 1
    all_that_idxs.remove(ex.that_idx)

    that_scores = []
    for that_idx in all_that_idxs:
        other_that_score = compute_single_local_affinity(s, ex.so_idx, that_idx, "so", "that")
        that_scores.append(other_that_score)

    okay_list = [x < correct_that_score for x in that_scores]
    return all(okay_list)

def get_num_correct(example_list: List[BaseExample]):
    ct_okay = 0
    for ex in example_list:
        if get_multithat_result(ex):
            ct_okay += 1
    return ct_okay

def exp2_multithat(**kwargs):
    # get the data
    exs = get_data()

    correct_ct = get_num_correct(exs)
    score = correct_ct / len(exs)

    return (score,)

