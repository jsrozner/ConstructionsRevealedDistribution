"""
This file for use with extracting results from mlm scoring on leonie corpus.

See also corpus_leonie.py which includes the original processing code to
create the dataset on which we ran the mlm scorer
"""
import math
from dataclasses import dataclass
from typing import List, Dict

from spacy.language import Language

from corpus_tools.zhou_cxs_so_difficult.corpus_leonie_eap_aap_cec import BaseExample
from lib.exp_common.mlm_result_for_sentence import MLMResultForSentence
from rozlib.libs.library_ext_utils.utils_spacy import get_child_with_dep


@dataclass
class ResultWrapper:
    orig_example: BaseExample
    mlm_result: MLMResultForSentence
    has_error: bool = False

def _align_mlm_results_with_input_data(
        mlm_results: List[MLMResultForSentence],
        all_exs: List[BaseExample]
):
    """
     since some of the inputs threw errors, they are not included in the output jsonl
     so we need to find the correct idx
    """
    miss_ct = 0
    aligned: List[int] = []
    orig_corpus_idx = 0
    i = 0

    # we are dealing only with every 5th entry (the ones that have so and that; no mods)
    # (it turns out that for any set of 5 there were either all errors, or no errors)
    # so we only need to align every fifth and that can be used for all the others in the set
    while i < len(mlm_results):
        if i%5 != 0:
            i+=1
            continue

        res = mlm_results[i]
        ex = all_exs[orig_corpus_idx]
        if res.sentence != ex.get_computed_sent_of_type(0):
            print(f"{res.sentence}")
            print(f"missing {ex.get_computed_sent_of_type(0)}")
            miss_ct += 1
            orig_corpus_idx += 1
            continue

        aligned.append(orig_corpus_idx)
        i += 1
        orig_corpus_idx += 1

    print(f"total missing: {miss_ct}")

    return aligned

def align_mlm_results_with_input_data(
        mlm_results: List[MLMResultForSentence],
        all_exs: List[BaseExample]
) -> List[ResultWrapper]:
    """
    Return a list of aligned results:
    For every entry in mlm_result, finds the appropriate input BaseExample
    """
    all_aligned_results = []
    aligned = _align_mlm_results_with_input_data(mlm_results, all_exs)

    total_err_ct = 0
    # todo: bug note that we have some errors here now
    # it seems that when we originally generated the cluster data,
    # that some of perturbed sentences were wrong. fixed some of
    # e.g., mlm_results[67] deletes a "the" instead of a "that"
    # we assert that this happens only for the perturbed examples
    for idx, res in enumerate(mlm_results):
        has_error = False
        base_idx = math.floor(idx/ 5)
        sent_type = idx % 5

        orig_ex_idx = aligned[base_idx]
        orig_ex = all_exs[orig_ex_idx]
        if not mlm_results[idx].sentence == orig_ex.get_computed_sent_of_type(sent_type):
            # print(f"checking:\n\t"
            #       f"{orig_ex.get_computed_sent_of_type(sent_type)}"
            #       f"\n\t{mlm_results[idx].sentence}")
            # print(orig_ex)
            has_error = True
            total_err_ct += 1
            if idx % 5 == 0:
                # unperturbed sentences shouldn't have these issues
                raise Exception("unmatched for unperturbed sent; this shouldn't happen")
        all_aligned_results.append(
            ResultWrapper(
                orig_ex,
                res,
                has_error=has_error
            )
        )

    print(f"Total errors (perturbed examples only; can ignore) (likely due to previous mis-indexing of "
          f"so/that/adj): {total_err_ct}")

    return all_aligned_results


def _align_example_with_spacy_parse(
        nlp: Language,
        base_ex: BaseExample):
    """
    Checks so, that, and adj indices
    Returns two maps:
     - one from the original word idxs (split on space) to the spacy indices (more splits)
     - and one from spacy index back to orig word idx
    """
    # Split the original text based on whitespace
    text = base_ex.sentence_punct_fixed
    original_words_from_ws_split = text.split()
    doc = nlp(base_ex.sentence_punct_fixed)

    # Create a mapping from original tokens to spaCy tokens
    ws_split_to_spacy_map = []    # will map from a given word in ws split to the idx in the spacy doc
    spacy_to_ws_map: Dict[int, int] = {}    # will map from a given word in ws split to the idx in the spacy doc
    start_idx = 0
    # print(text)
    for original_idx, word in enumerate(original_words_from_ws_split):
        # print(f"aligning {word}")
        # Find the first spaCy token whose character offset matches the start of the token
        did_find = False
        for spacy_word_idx, spacy_token in enumerate(doc):
            assert spacy_word_idx == spacy_token.i
            # print(f"checking {spacy_token}")
            tgt_idx = text.find(word, start_idx)
            if tgt_idx == -1:
                raise Exception(f"{word} not found starting at {start_idx}")
            # print(f"tgt_idx is {tgt_idx} and spacy idx is {spacy_token.idx}")
            if spacy_token.idx == tgt_idx:
                # start_idx = spacy_token.idx + 1
                start_idx = spacy_token.idx + len(spacy_token.text)
                ws_split_to_spacy_map.append(spacy_word_idx)

                spacy_to_ws_map[spacy_token.i] = original_idx

                # mapping.append((token, spacy_token))
                did_find = True
                break

        if not did_find:
            raise Exception(f"not found: {word}\n\t{text}")

    def print_err(idx, word):
        # if doc[mapping[idx]].text != "so":
        print(f"{word} not matching")
        print(base_ex.sentence_punct_fixed)
        if base_ex.sentence_punct_fixed.split(" ")[idx] != word:
            print(f"{base_ex.sentence_punct_fixed.split(" ")[idx]} is not {word}")
        print(idx, ws_split_to_spacy_map[idx])
        print(doc[ws_split_to_spacy_map[idx]])
        print([t for t in doc])
        raise Exception()

    # note these checks are partially duplicated below
    assert doc[ws_split_to_spacy_map[base_ex.so_idx]].text == "so", print_err(base_ex.so_idx, "so")
    assert doc[ws_split_to_spacy_map[base_ex.adj_idx]].text == base_ex.adj, print_err(base_ex.adj_idx, base_ex.adj)
    assert doc[ws_split_to_spacy_map[base_ex.that_idx]].text == "that", print_err(base_ex.that_idx, "that")

    return ws_split_to_spacy_map, spacy_to_ws_map

def _check_adj(nlp: Language, ex: BaseExample, aligned, idx):
    doc = nlp(ex.sentence_punct_fixed)
    tok = doc[aligned[idx]]
    pos = tok.pos_
    dep = tok.dep_
    if (pos != 'ADJ' and pos!='VERB') or dep != 'acomp':
        # otherwise error
        print(ex.sentence_punct_fixed)
        print(f"for word, {ex.sentence_punct_fixed.split(" ")[idx]}, {pos} not adj or verb or dep {dep} not acomp")
        raise Exception()

    # print(ex.sentence_punct_fixed)
    adj_head = tok.head
    nsubj = get_child_with_dep(adj_head, 'nsubj')
    if not nsubj:
        # e.g. "that working on X was so Y -> working"
        nsubj = get_child_with_dep(adj_head, 'csubj')
        print()
    if not nsubj:
        raise Exception("unable to parse subj for main clause")

    # print(f"adj head {adj_head} pos: {adj_head.pos_}")
    # print(f"nsubj: {nsubj}")

    # return verb and nsubj
    return adj_head, nsubj

def _check_that(nlp: Language, ex: BaseExample, aligned, idx):
    doc = nlp(ex.sentence_punct_fixed)
    tok = doc[aligned[idx]]
    pos = tok.pos_
    dep = tok.dep_
    if pos != 'SCONJ' or dep != 'mark':
        print(ex.sentence_punct_fixed)
        print(f"for word, {ex.sentence_punct_fixed.split(" ")[idx]}, {pos} not scong or {dep} not mark")
        raise Exception()

    # print(ex.sentence_punct_fixed)
    that_head = tok.head
    nsubj = get_child_with_dep(that_head, 'nsubj')
    if not nsubj:
        # e.g. there was
        nsubj = get_child_with_dep(that_head, 'expl')
    if not nsubj:
        raise Exception("unable to parse subj for complement clause")
    # print(f"that head {that_head} pos: {that_head.pos_}")
    # print(f"nsubj: {nsubj}")

    return that_head, nsubj

def _expect(nlp: Language, ex: BaseExample, aligned, idx, type):
    doc = nlp(ex.sentence_punct_fixed)
    pos = doc[aligned[idx]].pos_
    if not pos == type:
        print(ex.sentence_punct_fixed)
        print(f"for word, {ex.sentence_punct_fixed.split(" ")[idx]}, {pos} != {type}")
        raise Exception()

    return True

def get_other_pos_idxs(
        nlp: Language,
        base_ex: BaseExample,
        do_print = False
):
    orig_to_spacy_map, spacy_to_orig_map = _align_example_with_spacy_parse(nlp, base_ex)

    # checks (todo(low): i think we are repeating them - see above)
    _expect(nlp, base_ex, orig_to_spacy_map, base_ex.so_idx, 'ADV')
    _expect(nlp, base_ex, orig_to_spacy_map, base_ex.that_idx, 'SCONJ')
    # expect(ex, aligned, ex.adj_idx, 'ADJ')

    main_verb, main_subj = _check_adj(nlp, base_ex, orig_to_spacy_map, base_ex.adj_idx)
    clause_verb, clause_subj = _check_that(nlp, base_ex, orig_to_spacy_map, base_ex.that_idx)

    main_verb_idx = spacy_to_orig_map[main_verb.i]
    main_subj_idx = spacy_to_orig_map[main_subj.i]
    clause_verb_idx = spacy_to_orig_map[clause_verb.i]
    clause_subj_idx = spacy_to_orig_map[clause_subj.i]

    if do_print:
        words = base_ex.sentence_punct_fixed.split(" ")
        print(words[main_subj_idx], words[main_verb_idx],
              words[base_ex.so_idx], words[base_ex.adj_idx], words[base_ex.that_idx],
              words[clause_subj_idx], words[clause_verb_idx])

    return main_verb_idx, main_subj_idx, clause_verb_idx, clause_subj_idx

################
# post processing utils
################

# def read_mlm_results(use_orig_hhi) -> List[MLMResultForSentence]:
#     dir_root = Path("/Users/jsrozner/docs_local/_programming/research_constructions"
#                     "/constructions_repo")
#
#     if use_orig_hhi:
#         # original data with HHI
#         # mlm_data_dir = dir_root / "data_from_cluster/exp5_leonie_hhi/"
#         # mlm_file = mlm_data_dir/"corpus_leonie_all_exp5.jsonl"
#         mlm_file = "../../../../../data/zhou/corpus_leonie_all_??.jsonl"
#         # mlm_file = mlm_data_dir/"corpus_leonie_all_exp5.jsonl"
#     else:
#         # new data with surprisal (somehow the count is diff)
#         # note that this data does not have divergence scores; that may be the count difference since some would have failed
#         mlm_data_dir = dir_root / "data/output/leonie/exp5_leonie_cec_surprisal"
#         mlm_file = mlm_data_dir/"corpus_leonie_all.jsonl"
#
#     results = read_from_jsonl(mlm_file, MLMResultForSentence)
#     return results



