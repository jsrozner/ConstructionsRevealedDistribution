import spacy
from collections import Counter
from typing import List, Optional, Dict

from cxns_in_distrib.exp3_magpie.corpus_magpie import MLMResultForSentenceExp6, MAGPIE_entry, MAGPIE_Wrapper
from proj.cxs_are_revealed.paper.data_config import Exp3Magpie
from pprint import pp

from rozlib.libs.common.data.utils_jsonl import read_from_jsonl, read_jsonl


#########
# read in data functions
#########

# todo: don't know why but we had some dupes in 0.jsonl??
# todo (check whether there were dupes in 0.txt in the input?)
def get_all_magpie_results_unclean(
        data_dir = Exp3Magpie.magpie_affinity_dir
) -> List[MLMResultForSentenceExp6]:
    """
    Read in magpie results.

    "unclean" refers to the fact that we did not renormalize tokenization

    We had some duplicates in 0.jsonl, the code below removes the dupes
    """
    # dir_root = Path("/Users/jsrozner/docs_local/_programming/research_constructions/constructions_repo")
    # data_dir = dir_root / ("data_from_cluster/12_29_magpie_unclean")
    # data_dir = Exp3Magpie.magpie_affinity_dir
    print(data_dir)

    all_results = []
    all_ids = dict()
    for f in data_dir.glob("*.jsonl"):
        res: List[MLMResultForSentenceExp6] = read_from_jsonl(f, MLMResultForSentenceExp6)
        for r in res:
            if r.sentence_id in all_ids:
                assert all_ids[r.sentence_id] == r
                # print(f"duplicate {r.sentence_id} in {f} and {all_ids.get(r.sentence_id)}")
                continue
            else:
                all_ids[r.sentence_id] = r
                all_results.append(r)

    all_results = sorted(all_results, key=lambda x: x.sentence_id)
    return all_results

def load_minicons_score_data():
    """Read in the minicons scores for magpie data (ran on unclean data)"""
    magpie_scores = read_jsonl(Exp3Magpie.magpie_minicons_scores)
    all_magpie_scores = [x for x in magpie_scores]

    return all_magpie_scores

########
# POS helpers - so that we can identify N, verb, etc in magpie idiom words
#####
from tqdm import tqdm
from typing import Tuple
from spacy.tokens import Doc

def get_all_pos_for_magpie(magpie_entry_list: List[MAGPIE_entry]):
    nlp = spacy.load('en_core_web_sm')

    # to use transformer method
    # nlp = spacy.load('en_core_web_trf')
    # spacy.prefer_gpu()

    all_docs = []
    for me in tqdm(magpie_entry_list):
        doc = nlp(me.sent)
        all_docs.append(doc)
    return all_docs

## two helper functions to interact with the spacy docs to find the POS for a magpie entry
def get_pos_for_char_span(nlp_doc: Doc, char_span: Tuple[int, int]):
    for token in nlp_doc:
        if not token.idx == char_span[0]:
            continue
        if not token.idx + len(token) == char_span[1]:
            print("in get_pos_for_char_span - token length does not match")
            return None
        # print(f"found: {token}, {token.pos_}, {token.tag_}")
        return token.pos_

def get_all_pos_for_magpie_wrapper(nlp_doc: Doc, mag_wrapper: MAGPIE_Wrapper):
    for t in mag_wrapper.idiom_token_list:
        get_pos_for_char_span(nlp_doc, t.offset)

########
def check_data(magpie_wrappers, magpie_results: List[MLMResultForSentenceExp6]):
    """
    Verify alignment in data
    """
    print(len(magpie_wrappers), len(magpie_results))
    ct = 0
    fail_ct = 0
    for idx, (mag_wrapper, result) in enumerate(zip(magpie_wrappers, magpie_results)):
        # basic checks
        try:
            assert mag_wrapper.magpie_entry.id == result.sentence_id, \
                f"id no match {mag_wrapper.magpie_entry.id} != {result.sentence_id}"
            # assert idx == mag_wrapper.magpie_entry.id, f"index non match"   # we can use id or idx interchangeably
            assert mag_wrapper.magpie_entry.sent == result.sentence, f"sent non match"
            ct += 1
        except Exception as e:
            if fail_ct == 0:
                print(e)
                print(f"fail at idx {idx}; others won't be reported")
            fail_ct += 1
    if fail_ct > 0:
        print(f"had {fail_ct} fails; rest {ct} OK")
    else:
        print(f"check {ct}; all OK")

# todo: typing; move this and other new functions to exp3_analysis
def check_data2(magpie_wrappers, magpie_results: Dict[int, MLMResultForSentenceExp6]):
    """
    Verify alignment in data
    """
    print(len(magpie_wrappers), len(magpie_results.keys()))
    ct = 0
    fail_ct = 0
    for mag_wrapper in magpie_wrappers:
        # basic checks
        try:
            res = magpie_results[mag_wrapper.magpie_entry.id]
            assert mag_wrapper.magpie_entry.id == res.sentence_id, \
                f"id no match {mag_wrapper.magpie_entry.id} != {res.sentence_id}"
            # assert idx == mag_wrapper.magpie_entry.id, f"index non match"   # we can use id or idx interchangeably
            assert mag_wrapper.magpie_entry.sent == res.sentence, f"sent non match"
            ct += 1
        except Exception as e:
            if fail_ct == 0:
                print(e)
            fail_ct += 1
    if fail_ct > 0:
        print(f"had {fail_ct} fails; rest {ct} OK")
    else:
        print(f"check {ct}; all OK")
    return fail_ct

def filter_magpie(
        magpie_wrappers: List[MAGPIE_Wrapper],
        magpie_results: List[MLMResultForSentenceExp6],
        minicons_scores: Optional[List[dict]] = None,
        min_sent_length: Optional[int] = None,
        max_minicons_score: Optional[float] = None
) -> Tuple[List[MAGPIE_Wrapper], Counter]:
    # verify data is all okay
    check_data(magpie_wrappers, magpie_results)
    if max_minicons_score:
        assert minicons_scores and len(minicons_scores) == len(magpie_wrappers)

    log_counter = Counter()
    for x in ["error", "filtered_confidence", "filtered_short_sent"]:
        log_counter[x] = 1
    filtered: List[MAGPIE_Wrapper] = []
    for mag_wrapper in magpie_wrappers:
        idx = mag_wrapper.magpie_entry.id
        result = magpie_results[idx]
        if not result:
            log_counter['error'] += 1
            continue
        assert mag_wrapper.magpie_entry.sent == result.sentence, \
            f"{mag_wrapper.magpie_entry.sent} != {result.sentence}"
        #########
        # example-level exclusions
        if result.did_error:  # error in gpu calcs
            log_counter['error'] += 1
            continue

        if mag_wrapper.magpie_entry.confidence < 0.99:
            log_counter['filtered_confidence'] += 1
            continue

        if min_sent_length and len(mag_wrapper.magpie_entry.sent.split(" ")) < min_sent_length:
            log_counter['filtered_short_sent'] += 1
            continue

        if max_minicons_score:
            assert minicons_scores
            if minicons_scores[idx]['score'] < 0 or minicons_scores[idx]['score'] > max_minicons_score:
                log_counter['filtered_minicons'] += 1
                continue

        filtered.append(mag_wrapper)
    pp(log_counter)
    return filtered, log_counter

def filter_magpie2(
        magpie_wrappers: List[MAGPIE_Wrapper],
        magpie_results: Dict[int, MLMResultForSentenceExp6],
        minicons_scores: Optional[List[dict]] = None,
        min_sent_length: Optional[int] = None,
        max_minicons_score: Optional[float] = None
):
    # verify data is all okay
    check_data2(magpie_wrappers, magpie_results)
    if max_minicons_score:
        assert minicons_scores and len(minicons_scores) == len(magpie_wrappers)

    log_counter = Counter()
    for x in ["error", "filtered_confidence", "filtered_short_sent"]:
        log_counter[x] = 1
    filtered = []
    for mag_wrapper in magpie_wrappers:
        idx = mag_wrapper.magpie_entry.id
        result = magpie_results.get(idx)
        if not result:
            log_counter['error'] += 1
            continue
        assert mag_wrapper.magpie_entry.sent == result.sentence, \
            f"{mag_wrapper.magpie_entry.sent} != {result.sentence}"
        #########
        # example-level exclusions
        if result.did_error:  # error in gpu calcs
            log_counter['error'] += 1
            continue

        if mag_wrapper.magpie_entry.confidence < 0.99:
            log_counter['filtered_confidence'] += 1
            continue

        if min_sent_length and len(mag_wrapper.magpie_entry.sent.split(" ")) < min_sent_length:
            log_counter['filtered_short_sent'] += 1
            continue

        if max_minicons_score:
            assert minicons_scores
            if minicons_scores[idx]['score'] < 0 or minicons_scores[idx]['score'] > max_minicons_score:
                log_counter['filtered_minicons'] += 1
                continue

        filtered.append(mag_wrapper)
    pp(log_counter)
    return filtered, log_counter
