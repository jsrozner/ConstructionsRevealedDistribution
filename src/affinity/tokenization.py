import logging
import traceback
import warnings
from dataclasses import dataclass
from pprint import pp
from typing import List, Tuple, Set, Optional, Generator

import torch
from nltk import word_tokenize
from transformers import BatchEncoding, CharSpan

from lib.common.mlm_singleton import get_singleton_scorer
from lib.mlm_scorer import MLMScorer
from rozlib.libs.utils import string

_did_check_tokenizer_is_fast = False

class MultiTokenException(Exception):
    pass

def check_tokenizer_is_fast(mlm: MLMScorer):
    global _did_check_tokenizer_is_fast
    if _did_check_tokenizer_is_fast:
        return
    assert mlm.tokenizer.is_fast
    assert mlm.tokenizer.mask_token_id is not None
    _did_check_tokenizer_is_fast = True

Offset = Tuple[int, int]

########
# Utils
########

def clean_word(w: str) -> str:
    """Strips any punctuation from start or end of a word w"""
    start = 0
    end = len(w)
    for i in w:
        if i in string._punctuation:
            start += 1
        else:
            break
    for i in w[::-1]:
        if i in string._punctuation:
            end -= 1
        else:
            break
    return w[start: end]

def sentence_to_words_no_punct(sent: str) -> List[str]:
    """
    Returns a list of words with any starting or ending punctuation (in the words) stripped.
    For example, "I (saw ) you! 'hello', can't do-not" will return [I, saw, you, hello, can't, do-not]
    Args:
        sent:

    Returns: List of words with all punctuation at start or end of word stripped.
    Punctuation in the middle of a word will persist.
    """
    ret_list: List[str] = []
    for w in sent.strip().split(" "):
        clean_w = clean_word(w)
        if len(clean_w) > 0:
            ret_list.append(clean_w)
    return ret_list


def get_offsets_for_sent_and_clean_words(sent: str, clean_words: List[str]) -> List[Offset]:
    """
    Given a sentence and the clean words that were extracted from it,
    get the offsets (start, end) char positions where the words originally occurred
    Args:
        sent:
        clean_words:

    Returns:

    """
    ret_list: List[Offset] = []

    start = 0
    for w in clean_words:
        start = sent.find(w, start)
        assert start >= 0, f"{w} not found in {sent}"
        end = start + len(w)
        ret_list.append((start, start + len(w)))
        start = end

    return ret_list

########
@dataclass
class MaskedWord:
    """ A single masked word in a MultiMaskedSent"""
    masked_token_indices: List[int]
    original_tokens_ids: List[int]

@dataclass
class MultiMaskedSent:
    """
    Represents a Sentence's input_ids after masking masked_word_indices_in_sent

    masked_words and masked_word_indices_in_sent MUST line up
    """
    input_ids: torch.Tensor

    # if we multimask, then this stores which words in the word_list (sentence.words_clean)
    # were masked; ORDER MATTERS
    masked_words: List[MaskedWord]
    masked_word_indices_in_sent: List[int]

@dataclass
class MaskedSent:
    input_ids: torch.Tensor
    # todo: replace with MaskedWord / get rid of this and use MultiMaskedSent with single mask
    masked_token_indices: List[int]
    original_tokens_ids: List[int]


class Sentence:
    """
    Prepares a sentence for tokenization, so that we can just say "mask some word" or "mask some word index"
    """
    def __init__(
            self,
            sent: str,
            allow_non_alignment_in_tokenization = True,
            keep_punct = False
    ):
        """

        Args:
            sent:
            allow_non_alignment_in_tokenization: whether to only softly assert. Some models have punctuation added onto words
                This introduces potential issues if the non-alignment that we allow is something we are interested in checking.
                todo(imp) So we need to make sure that our get_inputs() functions in corr_matrix_new always assert that the target masked
                word is the correct one

        """
        if not allow_non_alignment_in_tokenization:
            warnings.warn(f"Allow alignment in tokenization is not set; might have a bad time with certain models"
                          f"YOU PROBABLY DON'T WANT THIS")
        mlm = get_singleton_scorer()
        check_tokenizer_is_fast(mlm)

        self.keep_punct = keep_punct
        self.sent = sent
        self.allow_nonalignment = allow_non_alignment_in_tokenization
        self.words_clean: List[str] = []
        self.word_clean_offsets: List[Offset] = []
        self.word_encodings: List[WordEncoding] = []

        # todo(imp): we probably want to discard inputs that are too long here in case seq length is too long
        #   - prob want consistent behavior
        #   - also seems to cause problems for some models (jdebene)
        self.encoding: BatchEncoding = mlm.get_batch_encoding_for_sentence(sent)

        # populate the above lists
        # todo(imp): maybe we should not except here; other callers should except bc Sentence will not behave properly if we catch here;
        #     will be hard to trace error
        try:
            self.fill_word_list()
        except Exception as e:
            traceback.print_exc()
            print(f"Failed to fill word list:\n\t{sent}; consider setting allow_non_alignment_in_tokenization, esp for other models")
            self.words_clean = self.word_clean_offsets = self.word_encodings = None


    def fill_word_list(self):
        if not self.keep_punct:
            self.words_clean = sentence_to_words_no_punct(self.sent)
        else:
            self.words_clean = word_tokenize(self.sent)
        self.word_clean_offsets = get_offsets_for_sent_and_clean_words(self.sent, self.words_clean)

        for i, offset in enumerate(self.word_clean_offsets):
            # internally this may error, but allow_nonalignment will cause it not to throw
            word = WordEncoding.make_word(i, self.sent, offset, self.encoding,
                                          allow_nonalignment=self.allow_nonalignment)

            # todo: this assert was useless bc word_chars is set by us to match the sentence
            # assert self.words_clean[i] == word.word_chars, f"word mismatch"
            self.word_encodings.append(word)

        assert len(self.word_encodings) == len(self.word_clean_offsets)

    def _get_inputs_with_masks_at_indices(
            self,
            indices: List[int]
    ) -> torch.Tensor:
        mlm = get_singleton_scorer()
        """
        Internal method to actually mask indices
        Args:
            indices:

        Returns:

        """
        # input_ids: torch.Tensor = self.encoding['input_ids']
        # todo(low): input_ids is not recognized here
        input_ids: torch.Tensor = self.encoding.input_ids
        input_ids = input_ids.clone()

        for idx in indices:
            # pyright note: asserted at top of file
            input_ids[0, idx] = mlm.tokenizer.mask_token_id  # pyright: ignore [reportArgumentType]

        return input_ids

    # todo(typing): tensor dimension
    # todo(imp): should deprecate this function and use the one below; note that this returns MaskedSent and other returns MultiMasked
    # currently we are keeping them both in sync
    def get_inputs_with_word_indices_masked(
            self,
            word_idxs_to_mask: List[int],
            allow_multi_token = False,
            expected_words: Optional[List[str]] = None
    ) -> MaskedSent:
        if expected_words and len(word_idxs_to_mask) != len(expected_words):
            raise Exception(f"len word idcs to mask does not match len expected words")
        indices = []
        # todo: this will be in the order of word_idxs passed in; not in the order of masks in the inputs
        original_tokens_ids: List[int] = []
        for i, word_idx in enumerate(word_idxs_to_mask):
            word_encoding = self.word_encodings[word_idx]
            if word_encoding.is_multiple_tokens and not allow_multi_token:
                raise MultiTokenException(f"Multitoken word {word_encoding.word_chars} but allow_multi_token not set")
            if word_encoding.has_potential_error:
                raise Exception("target mask has a potential error!")
                # logging.warning("target mask has a potential error!")
            if expected_words:
                expected_w = expected_words[i]
                # todo: check on lowercasing; add a param?
                if expected_w.lower() != word_encoding.word_chars.lower():
                    pp(self.word_encodings)
                    pp(self.words_clean)
                    pp(word_idxs_to_mask)
                    raise Exception(f"{expected_w} != {word_encoding.word_chars}")
                # assert expected_w == word_encoding.word_chars
            indices.extend(word_encoding.token_indices_in_tokenized_sent)
            original_tokens_ids.extend(word_encoding.token_ids)
        input_ids = self._get_inputs_with_masks_at_indices(indices)
        return MaskedSent(input_ids, indices, original_tokens_ids)

    def get_inputs_with_word_idx_masked(
            self,
            word_idx: int,
            allow_multi_token = False,
            expected_word: Optional[str] = None
    ) -> MaskedSent:
        """
        Single mask convenience function
        Args:
            word_idx:
            allow_multi_token:

        Returns:

        """
        expected = [expected_word] if expected_word else None
        return self.get_inputs_with_word_indices_masked([word_idx],
                                                        allow_multi_token,
                                                        expected_words=expected)

    def get_input_with_word_masked(
            self,
            word: str,
            occ: Optional[int] = None,
            allow_multi_token = False,
            lowercase=True
    ) -> MaskedSent:
        """
        Mask word, word, in sentence. If word occurs multiple times, occ (int, 0-indexed) must be passed to specify
        which occurrency to mask.

        Note that word and the sentence that are checked are both compared in lower case.

        Args:
            word:
            occ: 0-indexed
            allow_multi_token:

        Returns:

        """
        if not lowercase:
            raise NotImplementedError()

        # this treats all occurrences as lowercase
        word_idxs = [i for i, x in enumerate(self.words_clean) if x.lower() == word.lower()]
        if len(word_idxs) == 0:
            raise Exception(f"{word} not found in {self.sent}; words are {self.words_clean}")
        if len(word_idxs) == 1:
            assert occ is None or occ == 0, f"word occurs 1 time [{self.sent}]\nbut occ is {occ}"
            return self.get_inputs_with_word_idx_masked(
                word_idxs[0],
                allow_multi_token,
                expected_word=word
            )

        if occ is None:
            raise Exception(f"multiple instances of {word} found in {self.sent}; indicate occ")
        if occ >= len(word_idxs):
            raise Exception(f"Only {len(word_idxs)} occurrences of {word} in {self.sent} but idx [{occ}] given")
        return self.get_inputs_with_word_idx_masked(
            word_idxs[occ],
            allow_multi_token,
            expected_word=word
        )

    def inputs_for_each_word(
            self,
            allow_multi_token = False
    ) -> Generator[MaskedSent, None, None]:
        """
        Iterate through this sentence object and yield MaskedSent for each clean_word masked
        Args:
            allow_multi_token:

        Returns:

        """
        for i, _ in enumerate(self.words_clean):
            yield self.get_inputs_with_word_idx_masked(i, allow_multi_token)

    # todo: add assert checks that the words match
    # todo: currently need to keep in sync with fcn above
    def get_inputs_with_word_indices_masked_multi_mask(
            self,
            word_idxs_to_mask: List[int],
            allow_multi_token = False
    ) -> MultiMaskedSent:
        """
        For use with Local Affinity calculations. Mask multiple indices and return MultiMaskedSent
        Args:
            word_idxs_to_mask:
            allow_multi_token:

        Returns:

        """
        warnings.warn("using a function that does not assert that words match")

        tok_indices_to_mask = []

        masked_words: List[MaskedWord] = []

        for word_idx in word_idxs_to_mask:
            word_encoding = self.word_encodings[word_idx]
            if word_encoding.is_multiple_tokens and not allow_multi_token:
                raise Exception(f"Multitoken word {word_encoding.word_chars} but allow_multi_token not set")
            if word_encoding.has_potential_error:
                raise Exception("target mask has a potential error!")
            tok_indices_to_mask.extend(word_encoding.token_indices_in_tokenized_sent)
            masked_words.append(
                MaskedWord(
                    word_encoding.token_indices_in_tokenized_sent,
                    word_encoding.token_ids
                )
            )
        input_ids = self._get_inputs_with_masks_at_indices(tok_indices_to_mask)
        return MultiMaskedSent(input_ids, masked_words, word_idxs_to_mask)

    def get_sent_with_substituted_word_at_idx(self, idx: int, sub: str):
        """
        Utility so that we can inspect a sentence under a perturbation
        Args:
            idx:
            sub:

        Returns:

        """
        start, end = self.word_clean_offsets[idx]
        ret_str = self.sent[:start] + sub + self.sent[end:]
        return ret_str

def soft_assert(
        val_to_check: bool,
        error_msg,
        no_assert=False,
        do_print=False
):
    # normal assert
    if not no_assert:
        assert val_to_check, error_msg
        return
    # if true do nothing
    if val_to_check:
        return
    # otherwise print the error, but don't raise any exception
    if do_print:
        logging.warning(f"ASSERT: {error_msg}")
    else:
        warnings.warn("there was a soft assert error but do_print is false; will not report")

# todo: note adapted from WrappedIdiomWord
@dataclass
class WordEncoding:
    """
    Corresponds to a single word within a sentence, e.g.,
    A walk down memory lane -> one of the words will be "memory"
    """
    word_idx_in_sent: int
    # The offset in the original sentence that corresponds to this word
    offset: Tuple[int, int]
    # The actual characters in the original sentence for this word
    word_chars: str

    # Indices of the tokens that make up this word, e.g [5,6]
    token_indices_in_tokenized_sent: List[int]
    # Actual characters in the tokens, e.g., [mem, ory]
    token_words: List[str]
    token_ids: List[int]

    is_multiple_tokens: bool

    # todo: not currently in use; consider using if we end up masking a word with a potential tokenization error?
    has_potential_error: bool = False

    @classmethod
    def _maybe_fix_space_in_char_span(
            cls,
            orig_sent: str,
            span: CharSpan
    ) -> Tuple[CharSpan, int]:
        """
        Args:
            orig_sent:
            span:

        Returns:
            Tuple(new span, new span start)

        """
        mlm = get_singleton_scorer()
        if not mlm.allow_space_in_tokenization_span:
            return span, span.start

        if orig_sent[span.start] == " ":
            return CharSpan(span.start + 1, span.end), span.start + 1

        warnings.warn(f"allow space in tokenization span, but no space; this is probably fine and seems to occur sometimes")
        return span, span.start

    @classmethod
    def make_word(
            cls,
            i: int,
            orig_sent: str,
            offset: Tuple[int, int],
            encoding: BatchEncoding,
            allow_nonalignment = False
    ):
        has_potential_error = False
        def ass(bool_to_check, err_str):
            err_str = f"In {orig_sent}, {orig_sent[offset[0]: offset[1]]}, error\n\t" + err_str
            soft_assert(bool_to_check, err_str, allow_nonalignment)
            global has_potential_error
            has_potential_error = True

        # print("making word: ", orig_sent[offset[0]: offset[1]])
        word_chars = orig_sent[offset[0]:offset[1]]

        # get the tokens involved in the offset
        # char_to_token takes a char idx in original string and gives the index of the token in the encoding
        tok_set: Set[int] = set()
        for char_idx_in_offset in range(offset[0], offset[1]):
            tok_set.add(encoding.char_to_token(char_idx_in_offset))
        toks: List[int] = sorted(list(tok_set))
        # print("printing tokens:", toks)

        token_ids = [encoding.input_ids[0, t_idx] for t_idx in toks]
        # print("print token ids:", token_ids)

        # check that tokens exact match - make sure start and
        # encoding.token_to_chars returns a char_span as (start, end)

        # deal with LTG bert different behavior
        char_span_first_token = encoding.token_to_chars(toks[0])
        char_span_first_token, new_start = cls._maybe_fix_space_in_char_span(orig_sent, char_span_first_token)

        starting_char_idx = char_span_first_token.start
        ending_char_idx = encoding.token_to_chars(toks[-1]).end

        starting_matches = offset[0] == starting_char_idx
        ending_matches = offset[1] == ending_char_idx
        def get_tok_words():
            tok_spans = [encoding.token_to_chars(x) for x in toks]
            tok_words = [orig_sent[x[0]:x[1]] for x in tok_spans]
            return tok_words
        # print(starting_matches, ending_matches)
        ass(starting_matches and ending_matches,
            f"word = {word_chars}; (start, end) = ({starting_char_idx}, {ending_char_idx}) != expected ({offset[0]}, {offset[1]})"
            f"\n\ttokens: {toks}"
            f"\n\ttoken_ids: {token_ids}"
            f"\n\ttoken_words: {get_tok_words()}"
            )
        # assert starting_matches and ending_matches, \
        #     f"word = {word_chars}; (start, end) = ({starting_char_idx}, {ending_char_idx}) != ({offset[0]}, {offset[1]})"
        # if not starting_matches and ending_matches:
        #     f"word = {word_chars}; (start, end) = ({starting_char_idx}, {ending_char_idx}) != ({offset[0]}, {offset[1]})"

        # fill in tok chars
        tok_words = []
        for t in toks:
            char_span = encoding.token_to_chars(t)
            char_span, new_start = cls._maybe_fix_space_in_char_span(orig_sent, char_span)
            tok_words.append(orig_sent[char_span.start: char_span.end])
        total_token_chars = sum(map(lambda x: len(x), tok_words))
        len_matches = total_token_chars == offset[1] - offset[0]
        # assert len_matches
        ass(len_matches, "Length does not match")

        ass(len(tok_words) == len(token_ids) == len(toks), "")
        ass(word_chars == "".join(tok_words),"")

        # want a real assert on this one?
        # assert word_chars == "".join(tok_words)
        # assert len(tok_words) == len(token_ids) == len(toks)
        # assert word_chars == "".join(tok_words)

        wi = cls(
            i,
            offset,
            word_chars,
            toks,
            tok_words,
            token_ids,
            len(toks) > 1,
            has_potential_error=has_potential_error
        )
        return wi
