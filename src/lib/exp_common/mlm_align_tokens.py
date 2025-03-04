"""
This might be re-implementing the huggingface fill-mask pipeline
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Callable, Any

from transformers import RobertaTokenizer

from rozlib.libs.utils.string import str_is_mixed, is_english_alphanumeric
from rozlib.libs.library_ext_utils.utils_transformer import ROBERTA_SPACE_START_CHAR

# todo: note that this file is particular to a given tokenization scheme
# todo: we should be setting this from some sort of config, even if we use a module
# we could put this all in an "MLM module" that can then have the tokenizer be set in a single main
# we do this so that we don't have to pass around a ton of stuff in creating Tokenized word in sentence
# but note that the calls to decode here are sort of not pretty
mlm_align_tokens_tokenizer= RobertaTokenizer.from_pretrained('roberta-large')

# todo: check if punct with a word

@dataclass
class TokenizedWordInSentence:
    """
    For a given word with
    - str_rep (string rep of the word)
    - tokens is the list of the model's tokens (strings, not ids)
    - tok_idx_start - where in the overall token list for a sentence this word starts
    """
    str_rep: str
    # str_rep_no_special: str -- as a computed property

    tokens: List[str]   # token rep, potentially with funky representation
    # this is the overall token index *in the sentence* where this word starts
    tok_idx_start: int

    tokens_as_strings: List[str] = field(default_factory=list) # token rep, as a normal string, but potentially with spaces, e.g.,

    def __post_init__(self):
        if len(self.tokens) == 0:
            self.tokens_as_strings = []
        else:
            t_as_string = mlm_align_tokens_tokenizer.convert_tokens_to_string(self.tokens[0])
            self.tokens_as_strings = [t_as_string]

    @classmethod
    def from_token(cls, t: str, idx: int) -> TokenizedWordInSentence:
        # todo(low): think about this comment
        # note that we presume that the special token occurs only at beg, not in middle
        return TokenizedWordInSentence(t, [t], idx)

    def extend_by_token(self, t: str):
        t_as_string = mlm_align_tokens_tokenizer.convert_tokens_to_string(t)
        self.str_rep += t
        self.tokens.append(t)
        self.tokens_as_strings.append(t_as_string)

    @property
    def str_rep_no_special(self):
        return self.str_rep.strip(ROBERTA_SPACE_START_CHAR)

    @property
    def str_rep_as_string_no_space(self):
        # todo(low): hopefully this function is called only when we have a single elemnt in the list?

        # join the tokens as strings and strip any (generally preceding?) whitespace
        return "".join(self.tokens_as_strings).strip()


def reassembled_words_from_tokens_roberta(
        tok_list: List[str]
) -> List[TokenizedWordInSentence]:
    """
    Reassemble words based on how the tokenizer splits them
    """
    # - word
    # - idx
    all_words: List[TokenizedWordInSentence] = []
    curr_word = TokenizedWordInSentence("", [], 0)   # first word has no space
    did_end = False
    for idx, t in enumerate(tok_list):
        assert not did_end, f"Got token {t} after EOS"  # don't allow multiple EOS </s> tokens
        is_mixed = str_is_mixed(t)

        # BOS/ EOS / MASK handled specially
        if t == "<s>":
            assert idx == 0
            all_words.append(TokenizedWordInSentence(t, [t], idx))
            curr_word = TokenizedWordInSentence("", [], idx + 1)   # need to increment index since the word starts on next token
        elif t == "</s>":
            did_end = True
            all_words.append(curr_word)
            all_words.append(TokenizedWordInSentence(t, [t], idx))
        elif t == "<mask>":
            all_words.append(curr_word)
            all_words.append(TokenizedWordInSentence(t, [t], idx))
            curr_word = TokenizedWordInSentence("", [], idx + 1)   # need to increment index since the word starts on next token

        # space char handled specially - need to record prev word and start new one
        # todo: it would be better to use a while loop and then increment where we are in the string
        # then we could split tokens into multiple words as appropriate
        elif t.startswith(ROBERTA_SPACE_START_CHAR):
            # new word (got a space), so record the old one
            all_words.append(curr_word)
            # but if it starts with punctuation, then we mark it as its own "word" (avoid, e.g., "(known")
            # sometimes words are only a single char!
            if len(t) == 1 or not is_english_alphanumeric(t[1]):
                # todo: we should check that the rest of the token is also not alphanumeric
                # add the punctuation as its own word
                all_words.append(TokenizedWordInSentence.from_token(t, idx))
                curr_word = TokenizedWordInSentence("", [], idx + 1)   # need to increment index since word starts next token
            # otherwise, it's a normal word
            else:
                curr_word = TokenizedWordInSentence.from_token(t, idx)

        # has numeric/alpha/punctuation but likely occurs in middle of word so we keep
        # e.g. "I'm" -> I, 'm
        elif is_mixed:
            curr_word.extend_by_token(t)

        # todo: we should probably be operating on the string repr, but there's no one-to-one guarantee
        elif not is_english_alphanumeric(t[0]):    # not valid english alphabet and not mixed (ie some kind of punct)
            # if it's punct only and the start of a word, mark it as its own
            # if we are processing a word that is punct only, we don't want words to start with punct
            # note that second condition might never be used
            if len(curr_word.tokens) == 0 or not is_english_alphanumeric(curr_word.str_rep_as_string_no_space[0]):
                all_words.append(curr_word)
                all_words.append(TokenizedWordInSentence.from_token(t, idx))
                curr_word = TokenizedWordInSentence("", [], idx + 1)   # need to increment index since word starts next token

            # if we're in a word but get a punctuation that's not mixed, then split
            # if it's punctuation on its own, then end the current word
            # we only allow a single punctuation mark within a word

            elif not is_english_alphanumeric(tok_list[idx+1][0]):
                # print(f"nonalnum: Appending {t} at idx {idx}")
                all_words.append(curr_word)
                all_words.append(TokenizedWordInSentence.from_token(t, idx))
                curr_word = TokenizedWordInSentence("", [], idx + 1)   # need to increment index since word starts next token

            # todo(low): remove - this is covered by above case
            # # look ahead (e.g. GFOR D ' S
            # # not in the middle (e.g. it's not tok (non-alpha) tok; instead it's tok (non-alpha) new-word)
            # # todo(low): shoudl think about which tokens are allowed to be in the middle of words
            # elif tok_list[idx + 1][0] == ROBERTA_SPACE_START_CHAR or tok_list[idx + 1] == "</s>":
            #     # starts with punct (and is not mixed so gets its own word spot)
            #     # new punct, so record the old one
            #     all_words.append(curr_word)
            #     curr_word = TokenizedWordInSentence.from_token(t, idx)

            else:
                # in this case next token does not start with space, so it's in the middle
                curr_word.extend_by_token(t)

        else:
            # not mixed and first char is not punct. It could be numeric/ alpha only, etc
            curr_word.extend_by_token(t)

    return all_words

def find_in_list(
        elt_to_find,
        l: List[Any],
        start_idx: int,
        comp_fcn: Callable[[Any, Any], bool]) -> int | None:
    idx = start_idx

    if elt_to_find == "mask":
        elt_to_find = "<mask>"
    while True:
        try:
            elt = l[idx]
        except IndexError:
            logging.error(f"{elt_to_find} not found")
            return None

        if comp_fcn(elt_to_find, elt):
            return idx
        idx += 1


def align_words_with_token_list(
        sent_words_list: List[str],
        tokenized_words: List[TokenizedWordInSentence]):
    """
    For each w in sent_words_list, find the index such that tokenized_words[
    idx] aligns with w
    """
    def comp_fcn(word: str, tokenized_word: TokenizedWordInSentence):
        if word == tokenized_word.str_rep_no_special:
            return True
        elif word == tokenized_word.str_rep_as_string_no_space:
            return True
        return False

    tok_word_idx = -1   # bc we always add one in the call so that it does not find the same result again
    idx_map: List[TokenizedWordInSentence] = []
    for w in sent_words_list:
        tok_word_idx = find_in_list(w, tokenized_words, tok_word_idx + 1, comp_fcn)
        if tok_word_idx is None:
            raise Exception(
                f"For {sent_words_list}"
                f"word {w} not found in {tokenized_words}")
            # idx_map.append(None)
            # continue
        idx_map.append(tokenized_words[tok_word_idx])

    return idx_map




