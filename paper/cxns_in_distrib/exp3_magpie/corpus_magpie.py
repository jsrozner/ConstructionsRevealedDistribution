import warnings
from pathlib import Path

import torch
from transformers import BatchEncoding

from proj.cxs_are_revealed.paper.data_config import Exp3Magpie
from lib.common.mlm_singleton import get_singleton_scorer
from typing import List, Optional, Protocol
from dataclasses import dataclass, field
from typing import Tuple

from rozlib.libs.common.data.utils_jsonl import read_jsonl

# Allow use of the method we developed for paper1
# New work should not use it
_allow_deprecated_method = False
# todo: maybe rename this
def set_allow_deprecated():
    warnings.warn(f"Will use paper1 corpus_magpie method wrap idiom using encoding; don't use this in new code")
    global _allow_deprecated_method
    _allow_deprecated_method = True


def get_word_at_span(sent: str, span_tuple: Tuple[int, int]) -> str:
    """
    Return the char span (as str) from sent using span_tuple.
    """
    o_start, o_end = tuple(span_tuple)
    return sent[o_start: o_end]

# simplified version of MLMResultForSentence
@dataclass
class MLMResultForSentenceExp6:
    """
    One of these dataclasses per sentence, stored in jsonl format
    """
    # file_id: str        # matches the file id, unique
    sentence_id: int    # which sentence in order

    sentence: str       # full sentence

    tokens: List[str]   # the tokens for the sentence

    # we recorded originally, I think, hhi and surprisal?
    scores: List[List[float]]

    # multi_tok_indices: List[int]
    did_error: bool

# todo: assert no spaces within the offsets

@dataclass
class MAGPIE_entry:
    confidence: float
    context: List[str]
    document_id: int
    id: int
    sentence_no: str
    judgment_count: int
    label: str
    label_distribution: dict
    offsets: List[Tuple[int, int]]
    idiom: str
    split: str

    sent: str = field(init=False)

    def _check_idiom_and_offsets(self):
        """
        Verify that the expected idiom corresponds to the sentence words marked by offset
        """
        idiom_words = self.idiom.lower().split(" ")
        for o in self.offsets:
            w = get_word_at_span(self.sent, o)
            if w.lower() in idiom_words:
                return

        # print(f"{self.sent}\n not does not have any  {idiom_words} at offsetes")
        # raise Exception()

    def __post_init__(self):
        # context is always 5 sentences; we keep the one sentence that has the words we're interested in
        self.sent = self.context[2]
        # does not work - their labels are wrong
        # for o, w in zip(self.offsets, idiom_words):
        #     print(tuple(o), w)
        #     o_start, o_end = tuple(o)
        #     print(o_start, o_end)
        #     assert self.sent[o_start: o_end] == w, f"{self.sent[o_start: o_end]} != {w}"

        # check that the idiom at least overlaps some
        self._check_idiom_and_offsets()

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            d['confidence'],
            d['context'],
            d['document_id'],
            d['id'],
            d['sentence_no'],
            d['judgment_count'],
            d['label'],
            d['label_distribution'],
            [tuple(x) for x in d['offsets']],
            d['idiom'],
            d['split']
        )

class Wrappable(Protocol):
    id: int
    sent: str

@dataclass
class WrappedIdiomWord:
    """
    Corresponds to a single word within an idiom for a given MAGPIE entry, e.g.,
    A walk down memory lane -> one of the words will be "memory"

    Deprecated - use tokenization.py -> Sentence instead
    """
    # The offset in the original sentence that corresponds to this idiom word
    offset: Tuple[int, int]
    # The actual characters in the original Magpie sentence for this word
    idiom_word_chars: str

    # Indices of the tokens that make up this word, e.g [5,6]
    tokens: List[int]
    # Actual characters in the tokens, e.g., [mem, ory]
    token_words: List[str]

    is_multiple_tokens: bool
    tokens_do_not_exact_match: bool

    # todo: is contiguous (in original) - whether the idiom is split or not

    @classmethod
    def wrap_idiom_using_encoding(
            cls,
            magpie_entry: Wrappable,
            offset: Tuple[int, int],
            encoding: BatchEncoding,
    ):
        # todo: maybe don't want to mess up older files (will produce some error messages)
        if not _allow_deprecated_method:
            raise Exception("This method is deprecated. Avoid using it in new code."
                            "Use tokenization.py Sentence instead."
                            "If you're running code from paper1, call set_allow_deprecated() first"
                            "to use this function in paper1")
        # get the tokens involved in the offset
        toks = set()
        for char_idx_in_offset in range(offset[0], offset[1]):
            toks.add(encoding.char_to_token(char_idx_in_offset))
        toks = sorted(list(toks))

        # check if tokens exact match
        starting_char = encoding.token_to_chars(toks[0]).start
        ending_char = encoding.token_to_chars(toks[-1]).end
        starting_matches = offset[0] == starting_char
        ending_matches = offset[1] == ending_char

        idiom_chars = magpie_entry.sent[offset[0]: offset[1]]

        # fill in tok chars
        tok_words = []
        for t in toks:
            char_span = encoding.token_to_chars(t)
            tok_words.append(magpie_entry.sent[char_span.start: char_span.end])
        total_token_chars = sum(map(lambda x: len(x), tok_words))

        len_matches = total_token_chars == offset[1] - offset[0]

        tokens_match = starting_matches and ending_matches and len_matches
        if not tokens_match:
            print(magpie_entry.id)

        wi = cls(
            offset,
            idiom_chars,
            toks,
            tok_words,
            len(toks) > 1,
            not tokens_match
        )
        return wi


class MAGPIE_Wrapper:
    """
    Wraps a magpie entry with extra sentence info
    """
    def __init__(self, magpie_entry: MAGPIE_entry):
        """
        Steps:
        - get the total wordlist (contiguous non space strings)
        - get the alphabetic wordlist (candidates for masking)
        - get the list of idiom words (those with offsets)
            (checks that idiom words do not have spaces in them)

        """
        mlm = get_singleton_scorer()
        self.magpie_entry = magpie_entry

        encoding = mlm.get_batch_encoding_for_sentence(magpie_entry.sent)

        # idiom token indices
        self.idiom_token_list: List[WrappedIdiomWord] = self.get_idiom_token_list(encoding)

        # later (other methods)
        # produce a list of possible tokens / words to mask from a sentence

    def get_idiom_token_list(self, encoding: BatchEncoding):
        offset_to_token_list_map: List[WrappedIdiomWord] = []
        for offset in self.magpie_entry.offsets:
            wrapped_idiom = WrappedIdiomWord.wrap_idiom_using_encoding(self.magpie_entry, offset, encoding)
            offset_to_token_list_map.append(wrapped_idiom)

        assert len(offset_to_token_list_map) == len(self.magpie_entry.offsets)
        return offset_to_token_list_map

    # def make_token_list(self, encoding: BatchEncoding):
    #     tokens = encoding.tokens()
    #     # todo typing
    #     token_ids = encoding['input_ids'][0]
    #     special_tokens_mask = encoding['special_tokens_mask'][0]
    #     assert len(token_ids) == len(tokens)
    #     assert special_tokens_mask[0] == special_tokens_mask[-1] == 1
    #     assert all(special_tokens_mask[1:-2])
    #
    #     all_wrapped_tokens: List[TokenWrapper] = []
    #     for i, t in enumerate(tokens):
    #         token_chars = encoding.token_to_chars(i)
    #         all_wrapped_tokens.append(TokenWrapper(
    #             token_idx=i,
    #             token_id=token_ids[i],
    #             token_rep=t,
    #             char_span = token_chars
    #         ))
    #     return all_wrapped_tokens

class EntryForProcessing:
    """
    Takes a sentence and gets the tokens for masking.

    This is adapted from SentenceForMLMProcessing
    """
    def __init__(
            self,
            id: int,
            sent: str
    ):
        """
        Args:
            - example id
            - sent
        """
        mlm = get_singleton_scorer()
        self.sent_id = id
        self.sent = sent

        # todo: think about clenaing up tokenization spaces
        input_ids, tokens = mlm.prepare_inputs_for_sentence(sent)
        # needs to tokenize and return a list of tokens
        self.input_ids: torch.Tensor = input_ids
        self.tokens = tokens

    # get inputswithmaskatindex
    def get_inputs_with_mask_for_token_at_idx(
            self,
            tok_idx: int
    ) -> torch.Tensor:
        """
        Mask token at the index

        Args:
            word_idx: which tok in the sentence to mask

        """
        mlm = get_singleton_scorer()
        # clone, set mask, return input_ids
        input_ids = self.input_ids.clone()
        input_ids[0, tok_idx] = mlm.tokenizer.mask_token_id

        return input_ids

@dataclass
class MagpieEntryForProcessing:
    """Used for json to cluster"""
    id: int
    sent: str

#######
## magpie corpus helpers
#####

def get_all_magpie_json(f: Optional[Path] = None) -> List[dict]:
    """
    Read magpie from the corpus file into json
    """
    if f is None:
        f = Path(Exp3Magpie.original_magpie)
    j = read_jsonl(str(f))
    all_j = [x for x in j]

    # pp(all_j[15])
    return all_j

def get_magpie_pretty(
        magpie_json: List[dict]
) -> Tuple[List[MAGPIE_entry], List[MAGPIE_Wrapper]]:
    """
    For json parsed from magpie, return magpie entries and magpie wrappers

    """
    all_magpie_entries = []
    all_magpie_wrappers = []
    for idx, j in enumerate(magpie_json):
        m = MAGPIE_entry.from_dict(j)
        all_magpie_entries.append(m)
        mag_wrapper = MAGPIE_Wrapper(m)
        all_magpie_wrappers.append(mag_wrapper)
    return all_magpie_entries, all_magpie_wrappers
