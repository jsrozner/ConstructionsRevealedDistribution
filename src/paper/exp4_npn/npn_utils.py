from collections import Counter
from dataclasses import dataclass, field
from typing import List

@dataclass
class VocabItem:
    id: int
    token: str
    str_rep: str
    str_rep_no_space: str = field(init=False)

    def __post_init__(self):
        self.str_rep_no_space = self.str_rep.strip()

@dataclass
class NounRep:
    id: int
    token: str
    str_rep: str
    str_rep_no_space: str

@dataclass
class GPTOutput:
    noun: NounRep
    prep: str
    model: str
    output: str
    finish_reason: str

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

def filter_outputs(gpt_outputs: List[GPTOutput]):
    invalids = Counter()
    ret = []
    for o in gpt_outputs:
        # for 2 cats only
        # if o.prep != 'upon':
        #     continue
        if o.output.count(".") != 1:
            print('more than one period', o.output, "\n")
            invalids['period'] += 1
            continue
        if not o.output[0].isalpha() or not o.output[0].isupper():
            print('not upper case start', o.output, "\n")
            invalids['start'] += 1
            # continue
        tgt_str = f"{o.noun.str_rep_no_space} {o.prep} {o.noun.str_rep_no_space}"
        if o.output.lower().find(tgt_str) == -1:
            print('tgt str not found', tgt_str, o.output, "\n")
            invalids[o.prep] += 1
            continue
        if o.output.count(tgt_str) != 1:
            invalids['target_xtimes'] += 1
            print('tgt str more than once', tgt_str, o.output, "\n")
            continue
        if o.output.count(o.noun.str_rep_no_space) != 2:
            print('more than 2 occurrences!', o.noun.str_rep_no_space, o.output, "\n")
        ret.append(o)
    print(invalids)
    return ret
