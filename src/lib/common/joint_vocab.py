import warnings
from dataclasses import dataclass, field
from typing import List, Set

from transformers import PreTrainedTokenizerFast, AutoTokenizer

@dataclass
class VocabItem:
    id: int
    token: str
    str_rep: str
    str_rep_no_space: str = field(init=False)

    def __post_init__(self):
        self.str_rep_no_space = self.str_rep.strip()


def get_vocab(
        tokenizer: PreTrainedTokenizerFast,
        filter_likely_weird_ones = True
) -> List[VocabItem]:
    """
    Get all vocab items in the tokenizer vocab
    """
    print(tokenizer.name_or_path)
    vocab: List[VocabItem] = []
    for k,v in tokenizer.vocab.items():
        vocab.append(VocabItem(
            v, k, tokenizer.convert_tokens_to_string([k])
        ))

    # check to make sure it has a space preceding
    # this is sort of hacky and based on empirics
    # otherwise word probably does not work in the middle of sentence
    if filter_likely_weird_ones:
        warnings.warn(f"Will filter tokens that are likely not in the middle of string; bert not affected")
    else:
        warnings.warn(f"Will not filter tokens that are likely not in the middle of string; bert not affected")

    if tokenizer.name_or_path.startswith('google-bert'):
        # bert does not have this problem bc it uses the ## style for continuation
        return sorted(vocab, key=lambda x: x.id)

    if filter_likely_weird_ones:
        vocab = [
            x for x in vocab
            if len(x.token) > len(x.str_rep_no_space)
        ]

    return sorted(vocab, key=lambda x: x.id)
    # v = mlm_scorer.tokenizer.vocab
    # return [mlm_scorer.tokenizer.convert_tokens_to_string([s]) for s in v.keys()]
    # return vocab


def compute_joint_vocab(
        model_list: List[str],
        filter_likely_weird_ones = True
) -> Set[str]:
    """
    Compute joint vocab over all models in model_list (by checking each tokenizer).
    This was written for BabyLM work

    Args:
        model_list:
        filter_likely_weird_ones:

    Returns:

    """
    first_model = 0
    # print(model_list[first_model])
    tok = AutoTokenizer.from_pretrained(model_list[first_model], use_fast=True, clean_up_tokenization_spaces=True)
    _all_vocab = get_vocab(tok, filter_likely_weird_ones)

    all_vocab_str = [x.str_rep_no_space for x in _all_vocab]
    # all_vocab = get_vocab_clean(tok, strip_spaces=True)
    all_vocab = set(all_vocab_str)

    # num_nouns = len([x for x in all_vocab if is_valid_noun(x)])
    # print(f"num nouns: {num_nouns}")
    for model in model_list[first_model+1:]:
        # print(model)
        tok = AutoTokenizer.from_pretrained(model, use_fast=True, clean_up_tokenization_spaces=True)
        # print(len(tok.vocab.keys()))
        # vocab = set(tok.vocab.keys())
        # _vocab = get_vocab_clean(tok, strip_spaces=True)
        _vocab = get_vocab(tok)
        vocab = set([x.str_rep_no_space for x in _vocab])
        # to_remove = set(vocab).difference(all_vocab)
        # print("following are not in the new vocab")
        # to_remove = all_vocab.difference(vocab)
        # percent_loss = len(to_remove)/len(all_vocab)
        # print(f"percent loss: {percent_loss}")
        # print(list(to_remove)[:10])
        # prev_size = len(all_vocab)
        all_vocab.intersection_update(vocab)
        # num_nouns = len([x for x in all_vocab if is_valid_noun(x)])
        # print(f"new size: {len(all_vocab)}, estimated = {prev_size * (1-percent_loss)}")
        # print(f"num nouns: {num_nouns}")
    return all_vocab


