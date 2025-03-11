from __future__ import annotations

from typing import List, Set, Tuple

import torch

from lib.exp_common.mlm_gpu_common import TokenizationException
from lib.exp_common.mlm_align_tokens import TokenizedWordInSentence, reassembled_words_from_tokens_roberta, \
    align_words_with_token_list
from lib.common.mlm_singleton import get_singleton_scorer
from rozlib.libs.utils.string import split_and_remove_punct

# from libs.utils import split_and_remove_punct

mlm = get_singleton_scorer('roberta-large')

"""
Sentence Wrapper for use with GPU batch processing
"""
class SentenceForMLMProcessing:
    """
    Stores data for a sentence (tokenization, token alignment, etc) so that we don't have to recalculate wen
    preparing a batch
    """
    def __init__(
            self,
            file_id: str, # which file we're in
            sent_id: int,   # the idx in the file
            sent: str,
    ):
        # for logging / errors
        self.sent_file = file_id,
        self.sent_id = sent_id,

        self.sent = sent

        self.sent_input_ids: torch.Tensor

        # indexed by word index in sentence
        self.sent_words_list: List[str]
        self.aligned_word_reps: List[TokenizedWordInSentence]
        self.has_tokenization_issue: Set[int] = set()   # set of indices with tokenization issue

        self._prepare_sentence()

    def _prepare_sentence(self):
        """
        Called by __init__. Will produce
        - input_ids
        - sent_words_list
        - aligned_word_reps (for going from a word index to the tokens that correspond to that word)
        """
        # todo(low): note that as long as each batch contains only sentences of the same length we do not need padding

        # todo(low): encode and then ids_to_tokens repeats some logic
        # todo(low): we could also probably just squeeze self.sent_input_ids and then not call sent_input_ids[0] in the next line
        self.sent_input_ids = mlm.tokenizer.encode(self.sent, return_tensors='pt')  # pyright: ignore [reportUnknownMemberType]

        # we need to do alignment, so we go back to tokens
        tokens: List[str] = mlm.tokenizer.convert_ids_to_tokens(self.sent_input_ids[0])  # get 1st sent in batch
        # we need to check if each word is singly or multiply tokenized
        tokenized_words = reassembled_words_from_tokens_roberta(tokens)
        # align words with tokenization
        self.sent_words_list = split_and_remove_punct(self.sent)
        self.aligned_word_reps: List[TokenizedWordInSentence] = (
            align_words_with_token_list(self.sent_words_list, tokenized_words))
        assert len(self.sent_words_list) == len(self.aligned_word_reps)

    def _tokenization_exception(self, idx, error_msg):
        self.has_tokenization_issue.add(idx)
        msg_base = f"Tokenization exception in {self.sent_file}: {self.sent_id}\n{self.sent}"
        msg = msg_base + "\n\t" + error_msg
        raise TokenizationException(msg)

    def get_inputs_with_mask_for_word_at_idx(
            self,
            word_idx: int
    ) -> Tuple[torch.Tensor , int] | None:
        """
        Get input_ids and the token index with a mask for the word at word_dx

        Args:
            word_idx: which word in the sentence to mask

        Returns:
            Tuple
                - Tensor of inputs with mask at correct idx
                - idx of the token location with the mask (for selection later)
            None if word is multiply tokenized

        Excepts if unexpected result during tokenization
        """
        tgt_word = self.sent_words_list[word_idx]
        aligned_token_word = self.aligned_word_reps[word_idx]

        # check that tokenization is valid - invalid: either tokenization exception or multitokenized (return None)
        if not aligned_token_word.str_rep_no_special == tgt_word:
            self._tokenization_exception(word_idx, "Aligned word does not match target word")
        if len(aligned_token_word.tokens) > 1:
            self.has_tokenization_issue.add(word_idx)
            return None
        elif len(aligned_token_word.tokens) != 1:
            self._tokenization_exception(word_idx, "Token length is 0")

        # mask and check that we're masking in the right spot
        token_idx: int = aligned_token_word.tok_idx_start
        # Mask the current token (replace with [MASK])
        # todo(low): what is wrong with typing here (same as in mlm.py)
        # todo(low): this seems to be duplicating logic we already did (ie the same check)
        masked: int = self.sent_input_ids[0, token_idx]
        masked_word = mlm.tokenizer.decode([masked])  # pyright: ignore [reportUnknownMemberType]
        if masked_word.strip() != tgt_word:
            # todo(low): without strip we have an issue because spaces treated diff?
            self._tokenization_exception(word_idx,
                f"masked_word {masked_word}, not {tgt_word}")

        # clone, set mask, return input_ids
        input_ids = self.sent_input_ids.clone()
        input_ids[0, token_idx] = mlm.tokenizer.mask_token_id

        return input_ids, token_idx
