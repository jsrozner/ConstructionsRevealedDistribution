import statistics
import traceback
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pprint import pp
from typing import List, Dict

import nltk
from nltk.tokenize import word_tokenize

from affinity.corr_matrix_new import get_logits_for_masked_sent, compute_surprisal_for_logits
from affinity.tokenization import Sentence
from cxns_in_distrib.exp2_cogs.cogs_utils import CogsEntry, read_csv_row_by_row, get_all_data_clean
from proj.cxs_are_revealed.paper.data_config import Exp2Cogs
from lib.scoring_fns import probability
from rozlib.libs.utils.string import count_word_occurrences, get_nth_occ_string, get_nth_occ_list

# note subtle diff in spaces for two of the keys vs cogs_utils
tgt_keys_str_map = {
    'Let Alone': 'let alone',
    'Way Manner': 'way',
    'Conative': 'at',
    'Comparative Correlative': 'the the',
    'Much Less': 'much less',
    'Causative with CxN': 'with'
}

def check_nltk_tokenize():
    nltk.download('punkt')


@dataclass
class CogEntryTokenized(CogsEntry):
    """
    Will look like a CogsEntry, but ce.sent will be tokenized (extra spaces)
    - tgt_word_offsets (and tgt_word_occ_counts) will be updated to match the tokenization
    - tgt_words (indices) are kept the same
    """
    sent_orig: str = field(init=False)
    tgt_word_occ_counts: List[int] = field(init=False)

    @classmethod
    def make_from_cog_entry(cls, ce: CogsEntry):
        output = cls()
        output.id = ce.id
        output.cx_type = ce.cx_type
        output.sent_orig = ce.sent

        # this is not updated; it's the original word index
        output.tgt_words = list(ce.tgt_words)

        # these will be updated
        output.tgt_word_offsets = list(ce.tgt_word_offsets)
        output.sent = ce.sent   # this will be tokenized
        return output


def get_tokenized_cog_entry_for_cog_entry(cog_entry: CogsEntry) -> CogEntryTokenized:
    """
    For a single input entry, update it be tokenized. See description on CogEntryTokenized

    """
    # populate a new cog entry
    new_cog_entry = CogEntryTokenized.make_from_cog_entry(cog_entry)
    sent_tokenized_list = word_tokenize(new_cog_entry.sent)
    sent_tokenized_with_spaces = " ".join(sent_tokenized_list)   # this will be tokenized version (spaces bw punctuation)
    # we will populate these manually here
    new_cog_entry.sent = sent_tokenized_with_spaces
    # new_cog_entry.tgt_words = []
    new_cog_entry.tgt_word_offsets = []
    new_cog_entry.tgt_word_occ_counts = []

    # these are the cxn words we want to find
    tgt_cxn_words = tgt_keys_str_map[cog_entry.cx_type].split(" ")

    # one example has ways instead of way
    is_ways_exception = False

    # verify that the orig sentence matches offsets
    offsets = cog_entry.tgt_word_offsets
    for offset, t in zip(offsets, tgt_cxn_words):
        word = cog_entry.sent[offset[0]:offset[1]]
        if word.lower() != t.lower():
            assert word == "ways", f"in {cog_entry.sent} {word} != {t}"
            is_ways_exception = True

    # now we will populate based on re-tokenized sentence
    assert len(tgt_cxn_words) == len(offsets) == len(cog_entry.tgt_words), \
        print(cog_entry.sent,len(tgt_cxn_words), len(offsets), len(cog_entry.tgt_words))

    for o, target_word, tgt_word_idx in zip(offsets, tgt_cxn_words, cog_entry.tgt_words):
        if is_ways_exception:
            # change from way -> ways so everything works
            target_word = "ways"
        occ_idx_orig_str = count_word_occurrences(cog_entry.sent[:o[1]], target_word)
        assert occ_idx_orig_str >= 1

        idx_in_new_str = get_nth_occ_string(sent_tokenized_with_spaces.lower(), target_word.lower(), occ_idx_orig_str)
        sent_tokenized_lower_list = word_tokenize(sent_tokenized_with_spaces.lower())
        idx_in_word_list = get_nth_occ_list(sent_tokenized_lower_list, target_word, occ_idx_orig_str)

        new_offset = (idx_in_new_str, idx_in_new_str + len(target_word))
        new_word = sent_tokenized_with_spaces[new_offset[0]:new_offset[1]]
        assert new_word.lower() == target_word, f"{cog_entry.sent}, {new_word} != {target_word}"

        # record
        new_cog_entry.tgt_word_offsets.append(new_offset)
        # new_cog_entry.tgt_words.append(idx_in_word_list)
        # fix indexing 1 -> 0 indexing
        new_cog_entry.tgt_word_occ_counts.append(occ_idx_orig_str - 1)
    assert len(new_cog_entry.tgt_words) == len(new_cog_entry.tgt_word_offsets) == len(tgt_cxn_words) == len(new_cog_entry.tgt_word_occ_counts)
    return new_cog_entry

def get_tokenized_cogs_data() -> List[CogEntryTokenized]:
    csv_data_clean = read_csv_row_by_row(Exp2Cogs.cogs_parsed_final)
    cogs = get_all_data_clean(csv_data_clean)
    cogs_tokenized = [get_tokenized_cog_entry_for_cog_entry(x) for x in cogs]
    return cogs_tokenized

def get_affinity_for_cog_entry(ce: CogEntryTokenized) -> List[float]:
    s = Sentence(ce.sent, allow_non_alignment_in_tokenization=True)

    scores: List[float] = []
    for offset, occ in zip(ce.tgt_word_offsets, ce.tgt_word_occ_counts):
        expected_word = ce.sent[offset[0]: offset[1]]
        # print(expected_word, offset, occ)
        masked_sent = s.get_input_with_word_masked(
            expected_word,
            occ=occ,
            allow_multi_token=False
        )
        # obtain the logits at the masked position
        logits = get_logits_for_masked_sent(masked_sent)

        # compute probability of the original word
        prob = compute_surprisal_for_logits(
            masked_sent,
            logits,
            probability
        )
        scores.append(prob)
    return scores

def exp4_cogs(**kwargs):
    check_nltk_tokenize()

    # get the data
    cogs = get_tokenized_cogs_data()

    # aggregate scores
    # "much less" => two keys
    # "much less_much" and "much less_less"
    aggregator: Dict[str, List[float]] = defaultdict(list)
    ctr = Counter()
    for cog_entry in cogs:
        ctr["orig_" + cog_entry.cx_type] += 1
        try:
            scores = get_affinity_for_cog_entry(cog_entry)
        except Exception as e:
            traceback.print_exc()
            ctr["err_" + cog_entry.cx_type] += 1     # sent counter
            ctr["err_" + cog_entry.cx_type + " words"] += len(cog_entry.tgt_words)   # word ctr
            print(e)
            print(cog_entry)
            break
        cxn_words = tgt_keys_str_map[cog_entry.cx_type].split(" ")
        assert len(scores) == len(cxn_words)
        for cxn_word, score in zip(cxn_words, scores):
            key = f"{cog_entry.cx_type}_{cxn_word}"
            aggregator[key].append(score)

    output_dict: Dict[str, float] = {}
    for k in aggregator.keys():
        output_dict[k] = statistics.mean(aggregator[k])

    pp(output_dict)
    pp(ctr)

    return (
        *tuple(output_dict.values()),
    )

