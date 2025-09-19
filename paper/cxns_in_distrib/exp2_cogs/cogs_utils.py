from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple, List, Dict
import csv
import ast

import pandas as pd
from transformers import BatchEncoding

from cxns_in_distrib.exp3_magpie.corpus_magpie import WrappedIdiomWord
from lib.common.mlm_singleton import get_singleton_scorer



#####
# cogs csv utils
#####
@dataclass
class DataRow:
    cx_type: str = field(init=False)
    sentence: str = field(init=False)
    sentence_with_idxs: List[Tuple[int, str]] = field(init=False)
    tgt_words: List[int] = field(init=False)
    errors: List[str] = field(default_factory=list)

# todo: should go in csv utils; this was written for cogs though
def read_csv_row_by_row(file_path: str | Path) -> List[List[str]]:
    """Reads a CSV file row by row, ignoring the header.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        List[List[str]]: List of rows, where each row is a list of column values.
    """
    rows = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header row
        for row in reader:
            rows.append(row)
    return rows

def read_csv_by_column(file_path: str) -> Dict[str, List]:
    """Reads a CSV file and returns a dictionary where keys are column names
    and values are lists of column data.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        Dict[str, List]: Dictionary containing column-wise data.
    """
    df = pd.read_csv(file_path)
    return df.to_dict(orient="list")

def write_data_rows_to_csv(rows: List[DataRow], file_path: str) -> None:
    """Writes a list of DataRow instances to a CSV file.

    Args:
        rows (List[DataRow]): The list of DataRow instances.
        file_path (str): Path to save the CSV file.
    """
    # Convert dataclass instances to dictionaries
    data_dicts = [asdict(row) for row in rows]

    # Convert list fields to string for CSV storage
    for data in data_dicts:
        data["sentence_with_idxs"] = str(data["sentence_with_idxs"])
        data["tgt_words"] = str(data["tgt_words"])
        data["errors"] = ", ".join(data["errors"])  # Join error messages for readability

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(data_dicts)
    df.to_csv(file_path, index=False)

#####
# end csv utils
#####


#####
# cogs str utils
#####

def word_idx_to_offset(sentence: str, idx: int, tgt_word_len: int) -> Tuple[int, int]:
    pos = 0
    space_ct = 0
    assert sentence[0] != " "   # starting space would throw off indexing
    while space_ct < idx:
        if sentence[pos] == " ":
            space_ct += 1
        pos += 1
    start = pos
    end = pos + tgt_word_len
    return start, end

def count_symbols(s: str, symbols: List[str]) -> int:
    """Counts occurrences of specified symbols in a string.

    Args:
        s (str): The input string.
        symbols (Set[str]): A set of symbols to count.

    Returns:
    """
    symbols_set = set(symbols)
    ct = 0
    for sym in symbols_set:
        ct += s.count(sym)
    return ct

def get_end_of_str_by_punct(s: str):
    for sym in ['.', '!', '?']:
        idx = s.find(sym)
        if idx != -1: break

    return idx


#####
# end cogs str utils
#####

#####
# cogs initial cleaning
#####

def cogs_initial_clean(data_list: List[str], cx_type: str, tgt_words: str) -> List[DataRow]:
    """
    Used in cogs_preprocess.ipynb for initial cleaning

    Args:
        data_list:
        cx_type:
        tgt_words:

    Returns:

    """
    ret_list: List[DataRow] = []
    for s in data_list:
        # make sure valid string (some are weird or nans)
        if not isinstance(s, str):
            continue
        if len(s) < 4:
            continue

        # accrue info
        dr = DataRow()
        ret_list.append(dr)
        dr.cx_type = cx_type
        dr.sentence_with_idxs = [(idx, w) for idx, w in enumerate(s.split(" "))]

        if count_symbols(s, ['.', '!', '?']) != 1:
            dr.errors.append("no punct")    # this seems not to occur?
            dr.sentence = s
        else:
            idx = get_end_of_str_by_punct(s)
            truncated_str = s[:idx + 1]
            dr.sentence = truncated_str

        dr.tgt_words = []
        tgts = tgt_words.split(" ")
        for t in tgts:
            t_idxs = [idx for (idx, w) in dr.sentence_with_idxs if w.lower() == t.lower()]
            dr.tgt_words.extend(t_idxs)
            if s.lower().count(t) != 1:
                dr.errors.append("tgt word ct != 1")
    return ret_list


@dataclass
class CogsEntry:
    id: int = field(init=False)
    cx_type: str = field(init=False)
    sent: str = field(init=False)
    # indexes in the string
    # todo: rename to tgt_word_idcs
    tgt_words: List[int] = field(init=False)
    tgt_word_offsets: List[Tuple[int, int]] = field(default_factory=list)


"""Use our idioms code to extract HHI scores for each type"""
# adapted from mlm_gpu_exp6 and bnc_ruN_affinity

"""
- need to get char offsets -> done
- tokenize, then get token idx for offset
- mask that one, get score
"""
class CogEntryWrapper:
    """ Adapted from MAGPIE_Wrapper"""
    def __init__(self, cog_entry: CogsEntry):
        """
        Steps:
        - get the total wordlist (contiguous non space strings)
        - get the alphabetic wordlist (candidates for masking)
        - get the list of idiom words (those with offsets)
            (checks that idiom words do not have spaces in them)

        """
        mlm = get_singleton_scorer()
        self.cog_entry = cog_entry

        encoding = mlm.get_batch_encoding_for_sentence(cog_entry.sent)

        # idiom token indices
        self.idiom_token_list: List[WrappedIdiomWord] = self.get_idiom_token_list(encoding)

        # later (other methods)
        # produce a list of possible tokens / words to mask from a sentence

    def get_idiom_token_list(self, encoding: BatchEncoding):
        offset_to_token_list_map: List[WrappedIdiomWord] = []
        for offset in self.cog_entry.tgt_word_offsets:
            wrapped_idiom = WrappedIdiomWord.wrap_idiom_using_encoding(self.cog_entry, offset, encoding)
            offset_to_token_list_map.append(wrapped_idiom)

        assert len(offset_to_token_list_map) == len(self.cog_entry.tgt_word_offsets)
        return offset_to_token_list_map


tgt_keys_str_map_no_spaces = {
    'Let Alone': 'let alone',
    'Way Manner': 'way',
    'Conative': 'at',
    'Comparative Correlative': 'the the',
    'Much Less': 'much less',
    'Causative with CxN': 'with'
}


def get_all_data_clean(csv_data, fix_punct_in_comp_corr=False) -> List[CogsEntry]:
    """
    For use with data that was processed manually in google sheets

    `csv_data_clean = read_csv_row_by_row(Exp2Cogs.cogs_parsed_final)`
    `get_all_data_clean(csv_data_clean)`

    Args:
        csv_data:
        fix_punct_in_comp_corr:

    Returns:

    """
    ret_list: List[CogsEntry] = []
    ct_punct_changed = 0
    for idx, r in enumerate(csv_data):
        dr = CogsEntry()
        ret_list.append(dr)
        cx_type = r[0]
        if cx_type not in tgt_keys_str_map_no_spaces:
            raise Exception()
        dr.id = idx
        dr.cx_type = cx_type
        dr.sent = r[1]
        dr.tgt_words = ast.literal_eval(r[2])

        expected_words = tgt_keys_str_map_no_spaces[cx_type].split(" ")
        assert len(expected_words) == len(dr.tgt_words), f"{dr.sent} has {dr.tgt_words} != {expected_words}"
        for actual_word_idx, target_word in zip(dr.tgt_words, expected_words):
            actual_word = dr.sent.split(" ")[actual_word_idx].lower()
            if target_word.lower() != actual_word.lower():
                # one word is "ways" instead of "way"
                print(f"in sent {dr.sent}, at idx ({actual_word_idx}), {actual_word} != {target_word}")
                assert actual_word == "ways", f"{target_word}::{actual_word}"
                target_word = "ways"
            offset = word_idx_to_offset(dr.sent, actual_word_idx, len(target_word))
            w = dr.sent[offset[0]:offset[1]]
            assert w.lower() == target_word, f"{dr.sent}\n\t{offset}\t{w}\t{target_word}"
            dr.tgt_word_offsets.append(offset)

        # make sure that comparative correlative ends in a word without punctuation
        # otherwise sentences that end with COMP<.> will not be tokenized correctly
        if not fix_punct_in_comp_corr:
            continue
        if not dr.cx_type == 'Comparative Correlative':
            continue
        if not dr.sent.endswith("."):
            continue
        dr.sent = dr.sent[:-1] + " ."
        ct_punct_changed += 1

    print("note expected one error message bc one word is 'ways' instead of 'way'")
    if fix_punct_in_comp_corr:
        print(f"Moved {ct_punct_changed} periods at the end of sentence")

    return ret_list
