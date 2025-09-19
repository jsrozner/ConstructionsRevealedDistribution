import os.path
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from pprint import pp
from typing import List, Tuple, Dict

import pandas as pd

from rozlib.libs.common.data.utils_jsonl import read_from_jsonl


@dataclass
class NounRep:
    id: int
    token: str
    str_rep: str
    str_rep_no_space: str

@dataclass
class GPTOutput_with_nounrep:
    """ original data format; don't use this unless for original paper"""
    noun: NounRep
    prep: str
    model: str
    output: str
    finish_reason: str

    @property
    def npn(self):
        return f"{self.noun.str_rep_no_space} {self.prep} {self.noun.str_rep_no_space}"

@dataclass
class GPTOutput:
    noun: str
    prep: str
    model: str
    output: str
    finish_reason: str

    @property
    def npn(self):
        return f"{self.noun} {self.prep} {self.noun}"

@dataclass
class GPTResult:
    noun: str
    prep: str
    model: str
    output: str
    finish_reason: str

@dataclass
class HumanAnnotation:
    id: int
    sentence: str
    rating: int

def filter_outputs(
        gpt_outputs: List[GPTOutput],
        print_errors=False
) -> List[GPTOutput]:
    """
    Checks
    - has only one period
    - starts with capital
    - contains target string
    - contains target NPN string (okay to have case mismatch)
    - prints if a given noun is repeated more than twice
    Args:
        gpt_outputs:
        print_errors:

    Returns:

    """
    invalids = Counter()
    ret: List[GPTOutput] = []
    for o in gpt_outputs:
        if o.output.count(".") != 1:
            if print_errors:
                print('more than one period', o.output, "\n")
            invalids['period'] += 1
            continue
        if not o.output[0].isalpha() or not o.output[0].isupper():
            if print_errors:
                print('not upper case start', o.output, "\n")
            invalids['start'] += 1
            # continue
        # tgt_str = f"{o.noun.str_rep_no_space} {o.prep} {o.noun.str_rep_no_space}"
        tgt_str = f"{o.noun} {o.prep} {o.noun}"
        if o.output.lower().find(tgt_str) == -1:
            if print_errors:
                print('tgt str not found', tgt_str, o.output, "\n")
            invalids[o.prep] += 1
            continue
        if o.output.lower().count(tgt_str) != 1:
            if print_errors:
                print('tgt str more than once', tgt_str, o.output, "\n")
            invalids['target_xtimes'] += 1
            continue
        if o.output.count(o.noun) != 2:
        # if o.output.count(o.noun.str_rep_no_space) != 2:
            if print_errors:
                print('notify (non error): more than 2 occurrences!', o.noun, o.output, "\n")
                # print('more than 2 occurrences!', o.noun.str_rep_no_space, o.output, "\n")
        ret.append(o)
    print(f"In filter outputs, printing filtered")
    print(f"For the preposition category, it's count where desired NPN string did not occur")
    pp(invalids)
    return ret

def count_for_stats(
        judgements_aligned: List[HumanAnnotation],
        gpt_outputs_with_id: List[Tuple[int, GPTOutput]],
        min_rating = 4
):
    c = Counter()
    err_c = Counter()
    for j, e in zip(judgements_aligned, gpt_outputs_with_id):
        if j.sentence.lower().find('jew') != -1 or j.sentence.lower().find('heterosexual') != -1:
            err_c['bad word'] += 1
            continue
        if j.rating < min_rating:
            err_c[f"rating_{e[1].prep}"] += 1
            continue
        c[e[1].prep] += 1
    print("removed", err_c)
    print("totals after remove", c)
    return c

def count_for_stats_simple(gpt_outputs_after_filter):
    c = Counter()
    for g in gpt_outputs_after_filter:
        c[g.prep] += 1
    print(c)

def get_gpt_outputs_from_dir(
        dir: Path,
        print_errors=False
):
    gpt_outputs: List[GPTOutput] = read_from_jsonl(dir, GPTOutput)
    gpt_outputs_after = filter_outputs(gpt_outputs, print_errors=print_errors)
    return gpt_outputs_after

def write_data_rows_to_csv_for_human_analysis(
        entries: List[Tuple[int, GPTOutput]],
        file_path: Path
) -> None:
    # Convert dataclass instances to dictionaries
    # Convert list fields to string for CSV storage
    all_dicts: List[Dict] = []
    for e in entries:
        all_dicts.append({
            "id": e[0],
            "sent": e[1].output
        })

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(all_dicts)
    print(df.head(5))
    df.to_csv(file_path, index=False)

def make_csv_for_human_analysis(
        input_file_path: Path,
        output_file_path: Path,
        allow_overwrite = False
):
    gpt_outputs_after = get_gpt_outputs_from_dir(
        input_file_path,
        print_errors=True
    )

    print(f"after filtering total outputs: {len(gpt_outputs_after)}")

    # note does not have bad words removed yet
    count_for_stats_simple(gpt_outputs_after)

    gpt_outputs_with_id = [(idx, go) for idx, go in enumerate(gpt_outputs_after)]
    entries_randomized = [e for e in gpt_outputs_with_id]
    random.seed(42)
    random.shuffle(entries_randomized)

    if os.path.exists(output_file_path) and not allow_overwrite:
        raise Exception(f"{output_file_path} exists and allow_overwrite not set")

    write_data_rows_to_csv_for_human_analysis(entries_randomized, output_file_path)
