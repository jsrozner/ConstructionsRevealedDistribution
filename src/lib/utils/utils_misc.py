from typing import List, Callable

from rozlib.libs.utils.string import split_and_remove_punct


# from utils.string import split_and_remove_punct


# from libs.utils import split_and_remove_punct


def get_nth_occ(input_list: List[str], tgt: str, nth: int) -> int:
    """
    Returns the index of the nth occurrence of tgt in the input_list.
    Raises if (nth) occurrences of tgt are not found

    nth: 1-indexed
    """
    if nth <= 0:
        raise Exception("nth must be greater than 0 (it is 1 indexed)")

    ct = 0
    found_idx = -1  # search starts at found_idx + 1
    while ct < nth:
        # print(f"searching {input_list}from {found_idx + 1}")
        # will raise if not found
        found_idx = input_list.index(tgt, found_idx + 1)
        # print(f"found at {found_idx}")
        ct += 1

    return found_idx

def replace_word_with_substitution(
        sent: str,
        orig_word: str,
        word_idx: int,
        replace_word: str,
        do_print = False
) -> str:
    """
    For sent, replace orig_word (which should match the word list at that index) with
    replace_word.

    Will correctly handle repeated words
    """
    sent_word_list = split_and_remove_punct(sent)
    assert sent_word_list[word_idx] == orig_word

    # count to the word using spaces
    sent = " ".join(sent.split())   # make sure no double spaces?
    space_ct = 0
    idx = 0
    for idx, c in enumerate(sent):
        if space_ct == word_idx:
            break
        if c == ' ':
            space_ct += 1
            continue

    assert sent[idx:].startswith(orig_word), \
        f"{sent[idx:]} does not start with {orig_word}"

    # Replace the nth occurrence by slicing
    substituted_sent = (
            sent[:idx] + replace_word + sent[idx + len(orig_word):])

    if do_print:
        print(f"{sent}\n\t"
              f"after substit {orig_word}->{replace_word}\n"
              f"\t{substituted_sent}")

    return substituted_sent


# def test_nth_occ():
#     myl = [1, 2, 3, 1, 2, 3, 1, 2, 3]
#     assert get_nth_occ(myl, 1, 2) == 3


def fn_names(score_fns: List[Callable]):
    return [x.__name__ for x in score_fns]
