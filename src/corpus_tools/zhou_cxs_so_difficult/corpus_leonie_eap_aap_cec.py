from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List

import pandas as pd

from rozlib.libs.utils.string import is_any_alphanumeric, is_numeric


def get_clean_exs(path: Path, print_errors = False, minimal_clean=False) -> List[BaseExample]:
    """

    Args:
        path:
        print_errors:
        minimal_clean: newer approach that does not strip punctuation, etc.; it currently does not fill indicis of so,that,adj words (see below)

    Returns:

    """
    all_exs = get_all_exs(path,print_errors, minimal_clean=minimal_clean)
    print(f"Initially read in {len(all_exs)} examples")
    exs_no_errors = list(filter(lambda x: not x.has_error, all_exs))
    print(f"num exs after remove errors: {len(exs_no_errors)}")
    assert(all(map(lambda x: not x.has_error, exs_no_errors)))
    return exs_no_errors

def get_all_exs(file: Path, print_errors = False, minimal_clean=False):
    warnings.warn(f"in get_all_exs for leonie corpus; minimal clean is set to {str(minimal_clean).upper()}")
    data = pd.read_excel(file)
    all_exs = []
    err_ct = 0
    for idx, row in data.iterrows():
        # all_data.append()
        try:
            # change print_except to see what the issues are
            ex = BaseExample.from_data_row(idx, row, print_except=print_errors, minimal_clean=minimal_clean)
            all_exs.append(ex)
            if ex.has_error:
                err_ct +=1
        except Exception as e:
            err_ct += 1
            if print_errors:
                print(row['sentence'])
                print(e)
                print("\n\n")

    logging.warning(f"While processing, {err_ct} errors")
    return all_exs

class CxType(Enum):
    aap_causal = auto()     # note that causal == AAP
    eap_noncausal = auto()  # note that noncausal == EAP?
    cec = auto()
    oce = auto()
    invalid = auto()

    @classmethod
    def cx_type_from_label_string(cls, label: str) -> CxType:
        if label == "causal":
            return cls.aap_causal
        elif label == "causal excess":
            return cls.cec
        elif label == "non-causal":
            return cls.eap_noncausal
        elif label == "only causal excess":
            return cls.oce
        else:
            raise Exception(f"Unrecognized label {label}")

    @classmethod
    def name_for_value(cls, value: int):
        for member in cls:
            if member.value == value:
                return member.name
        raise ValueError(f"Value not found")

    @classmethod
    def pretty_name_for_value(cls, value: int):
        for member in cls:
            if member.value != value:
                continue
            if "_" in member.name:
                out = member.name.split("_")[0]
            else:
                out = member.name
            return out.upper()
        raise ValueError(f"Value not found")

    @classmethod
    def from_string(cls, s: str):
        """Converts something like 'CxType.X' to correct CxType"""
        if not s.startswith('CxType.'):
            raise ValueError
        label = s.split(".")[1]
        if label == "aap_causal":
            return cls.aap_causal
        elif label == "cec":
            return cls.cec
        elif label == "eap_noncausal":
            return cls.eap_noncausal
        elif label == "oce":
            return cls.oce
        else:
            raise Exception(f"Unrecognized label {label}")

    @classmethod
    def from_int(cls, label: int):
        """Converts something like 'CxType.X' to correct CxType"""
        if label == 1:
            return cls.aap_causal
        elif label == 2:
            return cls.eap_noncausal
        elif label == 3:
            return cls.cec
        elif label == 4:
            return cls.oce
        elif label == 5:
            return cls.invalid
        else:
            raise Exception(f"Unrecognized label {label}")


# @dataclass_json
@dataclass
class BaseExample:
    id: int
    adj: str
    label: CxType
    sentence_orig: str

    _sentence_punct_fixed: str | None
    has_error: bool
    multi_that: bool        # only valid to use if has_error is False

    so_idx: int
    adj_idx: int
    that_idx: int

    @classmethod
    def from_data_row(cls, idx: int, data: dict, print_except=False, minimal_clean=False):
        if minimal_clean:
            warnings.warn("the adj_idx, so_idx, and that_idx will be set to 0!")

        adj = data['adjective']
        cx_type = CxType.cx_type_from_label_string(data['label'])
        sent = data['sentence']

        # attempt to fix punctuation
        has_error = False
        try:
            if minimal_clean:
                sent_fixed = fix_punct_minimal(sent)
            else:
                sent_fixed = fix_punct(sent)
        except Exception as e:
            if print_except:
                print(e, "\n")
            has_error = True
            sent_fixed = None

        # get the indices of adj, so, that
        if sent_fixed:
            if minimal_clean:
                # todo: note that for these minimal examples, the indices are not usable
                adj_idx, so_idx, that_idx, multi_that = cls.parse_sent_minimal(
                    sent_fixed, adj, print_except)
            else:
                adj_idx, so_idx, that_idx, multi_that = cls.parse_sent(
                    sent_fixed, adj, print_except)
            if any(map(lambda x: x == -1, [adj_idx, so_idx, that_idx])):
                has_error = True
        else:
            assert has_error  # make sure it really is redundant
            has_error = True  # redundantly set; already set above if we're in else
            multi_that = False
            so_idx = adj_idx = that_idx = -1

        return BaseExample(
            idx,
            adj,
            cx_type,
            sent,
            sent_fixed,
            has_error,
            multi_that,
            so_idx,
            adj_idx,
            that_idx
        )

    @property
    def sentence_punct_fixed(self):
        if self._sentence_punct_fixed is None:
            raise Exception()
        return self._sentence_punct_fixed

    ####
    # methods to produce perturbed versions
    ####
    def _delete_words_at_indices(self, idx_list: List[int]):
        if self.has_error:
            raise Exception("invalid operation; this has an error")

        sent_words = self.sentence_punct_fixed.split(" ")
        keep_words = []
        for idx, w in enumerate(sent_words):
            if idx in idx_list:
                continue
            keep_words.append(w)
        return " ".join(keep_words)
    @property
    def o(self):
        return self.sentence_punct_fixed

    @property
    def ds(self):
        return self._delete_words_at_indices([self.so_idx])

    @property
    def dt(self):
        return self._delete_words_at_indices([self.that_idx])

    @property
    def dst(self):
        return self._delete_words_at_indices([self.so_idx, self.that_idx])

    @property
    def an(self):
        sent_words = self.sentence_punct_fixed.split(" ")

        sent_words.insert(self.so_idx, 'not')
        return " ".join(sent_words)

    def get_computed_sent_of_type(self, type:int):
        """
        Shorthand to access one of the o, ds, dt, dst, an values
        """
        # todo: this precomputes all of these?
        fns = [self.o, self.ds, self.dt, self.dst, self.an]
        return fns[type]


    @staticmethod
    def parse_sent(sent_fixed: str, adj: str, print_errors=False):
        """
        Locate the adjective as well as "so" and "that" in the sentence
        """
        sent_words = sent_fixed.split(" ")

        # todo: not an error, but we're going to be lazy; only 4 of these
        # expect the adjective only once
        adj_ct = sent_fixed.count(adj)
        if adj_ct != 1:
            if print_errors:
                print(f"{adj} repeated or not found in {sent_fixed}\n")
            return -1, -1, -1, False

        # get the adjective's position now
        try:
            adj_idx = sent_words.index(adj)
        except ValueError:
            if print_errors:
                print(f"Adj [{adj}] not found in sentence \n{sent_fixed}\n\n")
            return -1, -1, -1, False
        assert(sent_words[adj_idx] == adj)

        # calculate so position from adj position
        if sent_words[adj_idx - 1] == "so":
            so_idx = adj_idx - 1
            assert(sent_words[so_idx] == "so")
        else:
            if print_errors:
                print(f" so not found before adj {sent_fixed} [{sent_words[adj_idx -1]}]\n")
            so_idx = -1

        # calc that position
        multi_that = False

        # check multithat
        if sent_words[adj_idx + 1:].count('that') > 1:
            multi_that = True
            if print_errors:
                warnings.warn("Multiple thats - check multithat to see; we are linking to first that")
                # todo: using the first that will break, e.g., for
                # i was so certain that iw as right that i didn't plan for the alternative
                # but in the examples we saw, using the first that seemed always to be okay
                # print(f"Multiple thats; will use the first one after adj\n\t{sent_fixed}")

        if sent_words[adj_idx + 1:].count('that') == 0:
            that_idx = -1
            if print_errors: print(f"No that found\n\t{sent_fixed}\n")
        else:
            that_idx = sent_words[adj_idx + 1:].index('that') + adj_idx + 1

        if that_idx > -1:
            assert sent_words[that_idx] == "that", f"error in that: {sent_words[that_idx]} != that"

        return adj_idx, so_idx, that_idx, multi_that

    @staticmethod
    def parse_sent_minimal(sent_fixed: str, adj: str, print_errors=False):
        """
        Locate the adjective as well as "so" and "that" in the sentence

        If more than one "so" will error.
        """
        # expect the adjective only once
        adj_ct = sent_fixed.count(adj)
        if adj_ct != 1:
            if print_errors:
                print(f"WARN: {adj} repeated or not found ({adj_ct}) in {sent_fixed}\n")
            # we will keep repeated adjs
            # return -1, -1, -1, False

        # calculate so position from adj position
        so_ct = sent_fixed.count("so ")
        if so_ct == 0:
            if print_errors:
                print(f"ERROR so not found in {sent_fixed}\n")
            return -1, -1, -1, False
        if so_ct > 1:
            # todo: we can do pretty well just by checking if the word after "so" is the adj
            # but we have to keep track of which so to mask
            if print_errors:
                print(f"ERROR more than one so in {sent_fixed}\n")
            return -1, -1, -1, False

        # calc that position
        multi_that = False

        num_thats = sent_fixed.count("that")
        if num_thats == 0:
            if print_errors:
                print(f"ERROR that not found in {sent_fixed}\n")
            return -1, -1, -1, False
        if num_thats > 1:
            multi_that = True
            if print_errors:
                print("WARN: Multiple thats - check multithat to see; we are linking to first that")
                # todo: using the first that will break, e.g., for
                # i was so certain that iw as right that i didn't plan for the alternative
                # but in the examples we saw, using the first that seemed always to be okay
                # print(f"Multiple thats; will use the first one after adj\n\t{sent_fixed}")

        # todo: note that we are just returning nothing here
        return 0, 0, 0, multi_that

    # ####
    # ## serialization stuff
    # def to_dict(self):
    #     print('in todict')
    #     data = super().to_dict()
    #     data["label"] = self.label.value  # Convert Enum to value
    #     return data
    #
    # @classmethod
    # def from_dict(cls, data):
    #     data["label"] = CxType(data["label"])  # Convert value back to Enum
    #     return super().from_dict(data)

    def _parse_bool(self, bool_as_str: str) -> bool:
        """
        Parses a boolean that was stored as TRUE or FALSE string in csv
        Args:
            bool_as_str:

        Returns: bool

        """
        if bool_as_str == "TRUE":
            return True
        elif bool_as_str == "FALSE":
            return False
        else:
            raise Exception(f"Invalid for bool: {bool_as_str}")

    def _fix_self_for_csv_read(self):
        """
        When read in from CSV, all things will be strings
        """
        self.id = int(self.id)
        self.label = CxType.from_string(self.label)
        # note that sentence_orig may be wrong!
        self.has_error = self._parse_bool(self.has_error)
        self.multi_that = self._parse_bool(self.multi_that)
        self.so_idx = int(self.so_idx)
        self.adj_idx = int(self.adj_idx)
        self.that_idx = int(self.that_idx)


valid_punct = ["'", ',', ':', ';', '.', '!', '"', "(", ")", "-"]
def fix_punct(sent: str):
    """
    Cleans up all punctuation issues in the leonie corpus.
    """
    sent = re.sub(r'\(so\)', 'so', sent)
    sent = re.sub(r'\(that\)', 'that', sent)
    sent = re.sub(r'’', "'", sent)
    sent = re.sub(r'“', "'", sent)
    sent = re.sub(r'“', "”", sent)
    indxs_of_spaces_to_delete = []
    in_quote = False
    in_paren = False
    for i, c in enumerate(sent):
        if c in valid_punct:
            if c in ["-"]:
                # check consistency
                assert (sent[i-1] == ' ') == (sent[i+1] == ' ')
                if sent[i-1] == ' ':
                    indxs_of_spaces_to_delete.append(i-1)
                    indxs_of_spaces_to_delete.append(i+1)

            elif c in ["'"]:
                # special handle did n't / could n't
                if sent[i-1:i+2] == "n't":
                    assert sent[i-2] == ' ', f"{sent} punct issue 1"
                    indxs_of_spaces_to_delete.append(i-2)
                # normal contractions
                elif sent[i:i+2] in ["'s", "'m", "'r", "'d", "'v"]:
                    assert sent[i-1] == ' '
                    indxs_of_spaces_to_delete.append(i-1)
                # likely not split
                elif is_any_alphanumeric(sent[i-1]) and is_any_alphanumeric(sent[i+1]):
                    continue
                else:
                    raise Exception(f"unhandled apostrophe [{sent[i:i+1]}]\n\t{sent}")
            # if colon in the middle of a number
            elif c in [":"] and is_numeric(sent[i-1]) and is_numeric(sent[i+1]):
                continue
            elif c in [',', ":", ';']:
                assert sent[i-1] == ' ', f"{sent} punct issue 2"
                indxs_of_spaces_to_delete.append(i-1)
            elif c == '.':
                # can have, e.g., St. Francis
                if sent[i-1] == ' ':
                    indxs_of_spaces_to_delete.append(i-1)
            elif c == '!':  # no question marks
                assert sent[i-1] == ' ', f"{sent} punct issue 3"
                indxs_of_spaces_to_delete.append(i-1)
            # todo: single ' as quote is not handled
            elif c == '"':
                if not in_quote:
                    # this is a quote: " Hi "
                    assert sent[i+1] == ' '    # we could be at beg. of sent
                    indxs_of_spaces_to_delete.append(i+1)
                else:
                    assert sent[i-1] == ' ' # we could be at end of sentence
                    indxs_of_spaces_to_delete.append(i-1)
                in_quote = not in_quote
            elif c == '(':
                if in_paren:
                    raise Exception("double open paren")
                assert sent[i+1] == ' '
                indxs_of_spaces_to_delete.append(i+1)
                in_paren = True
            elif c == ')':
                if not in_paren:
                    raise Exception("unopened close paren")
                assert sent[i-1] == ' '
                indxs_of_spaces_to_delete.append(i-1)
                in_paren = False
            else:
                raise Exception(f"not handled {c}")
        elif not is_any_alphanumeric(c) and not c == ' ':
            raise Exception(f"Unrecognized {c} in {sent}")

    assert not in_quote, f"No terminating quote! in {sent}"

    final_sent = ""
    for idx, c in enumerate(sent):
        if not idx in indxs_of_spaces_to_delete:
            final_sent += c
        else:
            assert c == ' '

    return final_sent

def fix_punct_minimal(sent: str):
    """
    Cleans up all punctuation issues in the leonie corpus.
    """
    sent = re.sub(r'\(so\)', 'so', sent)
    sent = re.sub(r'\(that\)', 'that', sent)
    # sent = re.sub(r'’', "'", sent)
    # sent = re.sub(r'“', "'", sent)
    # sent = re.sub(r'“', "”", sent)

    return sent

####
# fix examples helpers

def find_in_example_list(ex_list: List[BaseExample], id: int) -> BaseExample:
    """
    Find example with id in the ex_list
    """
    e_list = [e for e in ex_list if e.id == id]
    # print(e_list)
    if not len(e_list) == 1:
        print(f"returning 0 or more than one example!!")
        # raise Exception("returning more than one example")
        # todo
        return e_list  # pyright: ignore [reportReturnType]

    # assert len(e_list) == 1
    return e_list[0]


def fix_labels(ex_list: List[BaseExample]):
    """
    Replace wrong labels using the label_changes defined in this file.
    """
    for change_list in [label_changes, label_changes2]:
        for id, new_label in change_list:
            ex = find_in_example_list(ex_list, id)
            if not isinstance(ex, BaseExample):
                print(f"WARN: ex not found")
                continue
            ex.label = CxType.from_int(new_label)


# list of the incorrectly labeled examples to potentially change labels
label_changes = [
    # This was so funny that I had to buy another copy and read it to my better half.; AAP -> CEC
    (63, 3),

    # omit
    # After a series of fires in 1741, the city became so panicked that blacks planned to burn the city in conspiracy with some poor whites.; AAP -> invalid
    (95, 5),

    # omit
    # In Burma, the belief was once so widespread that the Sumatran rhino ate fire.; EAP -> invalid
    (135, 5),

    #It's his lucky quarter and Pop feels so bad that Lucky lost it.
    # CEC -> AAP
    (10, 1),

    # I am so fortunate to have had it recommended to me so highly that I bought the eight pack.; CEC -> invalid
    (58, 5),

    # For example, Zhu Youliang () the Prince of Heng, an older cousin of the emperor's, was so honored at court at that time that even Li Yu's superior, the chief of staff Li Zhen, kneeled to him.
    # see below - pipeline may have had an error and recorded wrong score

    # I am so ashamed of myself that I ignored other reviewers and paid money for this book.
     # -> AAP
    (246, 1),
]
label_changes2 = [
    # "Seeing this in a jar, I was so hopeful that it had it's own well sealed jar protecting the powder from hardenin".
    # EAP -> invalid
    (286, 5),

    # 1: He was so afraid that rival loyalist inmates wished to kill him inside the prison.
    # EAP -> invalid
    (1, 5),


]

# for EMNLP UMAP redo
additional_ambiguous = [1, 320, 270, 95, 286]
all_changes = label_changes + label_changes2
ambiguous_labels = [x[0] for x in all_changes]
all_ambiguous_labels = ambiguous_labels + additional_ambiguous




# used for babyLM
def get_data(
        data_path: Path
) -> List[BaseExample]:
    # todo: most errors are double so; we can fix; see other file (see cec_suite.ipynb)
    # note that we use minimal_clean=False which does not indicate the adj/that/so indices
    # our code in this file only processes single "so" examples and then just masks that single word
    exs_no_errors = get_clean_exs(data_path, print_errors=False, minimal_clean=True)
    print("fixing labels")
    fix_labels(exs_no_errors)
    print(f"number of examples: {len(exs_no_errors)}")
    return exs_no_errors
