from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Generator

import torch
from transformers import BatchEncoding

from lib.common.mlm_singleton import get_singleton_scorer
from rozlib.libs.utils import string

mlm = get_singleton_scorer()
assert mlm.tokenizer.is_fast
assert mlm.tokenizer.mask_token_id is not None

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
    def __init__(self, sent: str):
        self.sent = sent
        self.words_clean: List[str] = []
        self.word_clean_offsets: List[Offset] = []
        self.word_encodings: List[WordEncoding] = []

        self.encoding: BatchEncoding = mlm.get_batch_encoding_for_sentence(sent)

        # populate the above lists
        self.fill_word_list()

    def fill_word_list(self):
        self.words_clean = sentence_to_words_no_punct(self.sent)
        self.word_clean_offsets = get_offsets_for_sent_and_clean_words(self.sent, self.words_clean)

        for offset in self.word_clean_offsets:
            word = WordEncoding.make_word(self.sent, offset, self.encoding)
            self.word_encodings.append(word)

        assert len(self.word_encodings) == len(self.word_clean_offsets)

    def _get_inputs_with_masks_at_indices(
            self,
            indices: List[int]
    ) -> torch.Tensor:
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
    def get_inputs_with_word_indices_masked(
            self,
            word_idxs_to_mask: List[int],
            allow_multi_token = False
    ) -> MaskedSent:
        indices = []
        # todo: this will be in the order of word_idxs passed in; not in the order of masks in the inputs
        # todo: should deprecate this function and us the one below
        original_tokens_ids: List[int] = []
        for word_idx in word_idxs_to_mask:
            word_encoding = self.word_encodings[word_idx]
            if word_encoding.is_multiple_tokens and not allow_multi_token:
                raise Exception(f"Multitoken word {word_encoding.word_chars} but allow_multi_token not set")
            indices.extend(word_encoding.token_indices_in_tokenized_sent)
            original_tokens_ids.extend(word_encoding.token_ids)
        input_ids = self._get_inputs_with_masks_at_indices(indices)
        return MaskedSent(input_ids, indices, original_tokens_ids)

    def get_inputs_with_word_idx_masked(
            self,
            word_idx: int,
            allow_multi_token = False
    ) -> MaskedSent:
        """
        Single mask convenience function
        Args:
            word_idx:
            allow_multi_token:

        Returns:

        """
        return self.get_inputs_with_word_indices_masked([word_idx], allow_multi_token)

    def get_input_with_word_masked(
            self,
            word: str,
            occ: Optional[int] = None,
            allow_multi_token = False
    ) -> MaskedSent:
        """
        Mask word, word, in sentence. If word occurs multiple times, occ (int, 0-indexed) must be passed to specify
        which occurrency to mask.

        Args:
            word:
            occ: 0-indexed
            allow_multi_token:

        Returns:

        """
        word_idxs = [i for i, x in enumerate(self.words_clean) if x == word]
        if len(word_idxs) == 0:
            raise Exception(f"{word} not found in {self.sent}.")
        if len(word_idxs) == 1:
            assert occ is None or occ == 0
            return self.get_inputs_with_word_idx_masked(word_idxs[0], allow_multi_token)

        if not occ:
            raise Exception(f"multiple instances of {word} found in {self.sent}; indicate occ")
        if occ >= len(word_idxs):
            raise Exception(f"Only {len(word_idxs)} occurrences of {word} in {self.sent} but idx [{occ}] given")
        return self.get_inputs_with_word_idx_masked(word_idxs[occ])

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
        tok_indices_to_mask = []

        masked_words: List[MaskedWord] = []

        for word_idx in word_idxs_to_mask:
            word_encoding = self.word_encodings[word_idx]
            if word_encoding.is_multiple_tokens and not allow_multi_token:
                raise Exception(f"Multitoken word {word_encoding.word_chars} but allow_multi_token not set")
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



# todo: adapted from WrappedIdiomWord
@dataclass
class WordEncoding:
    """
    Corresponds to a single word within a sentence, e.g.,
    A walk down memory lane -> one of the words will be "memory"
    """
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

    @classmethod
    def make_word(
            cls,
            orig_sent: str,
            offset: Tuple[int, int],
            encoding: BatchEncoding
    ):
        # we might not need to re-store this but whatever
        word_chars = orig_sent[offset[0]:offset[1]]

        # get the tokens involved in the offset
        # char_to_token takes a char idx in original string and gives the index of the token in the encoding
        tok_set: Set[int] = set()
        for char_idx_in_offset in range(offset[0], offset[1]):
            tok_set.add(encoding.char_to_token(char_idx_in_offset))
        toks: List[int] = sorted(list(tok_set))

        token_ids = [encoding.input_ids[0, t_idx] for t_idx in toks]

        # check that tokens exact match - make sure start and
        # encoding.token_to_chars returns a char_span as (start, end)
        starting_char = encoding.token_to_chars(toks[0]).start
        ending_char = encoding.token_to_chars(toks[-1]).end
        starting_matches = offset[0] == starting_char
        ending_matches = offset[1] == ending_char
        assert starting_matches and ending_matches

        # fill in tok chars
        tok_words = []
        for t in toks:
            char_span = encoding.token_to_chars(t)
            tok_words.append(orig_sent[char_span.start: char_span.end])
        total_token_chars = sum(map(lambda x: len(x), tok_words))
        len_matches = total_token_chars == offset[1] - offset[0]
        assert len_matches

        assert len(tok_words) == len(token_ids) == len(toks)
        assert word_chars == "".join(tok_words)

        wi = cls(
            offset,
            word_chars,
            toks,
            tok_words,
            token_ids,
            len(toks) > 1,
        )
        return wi
