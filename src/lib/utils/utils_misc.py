from pathlib import Path
from typing import List, Callable

from matplotlib.figure import Figure

from rozlib.libs.utils.string import split_and_remove_punct
from rozlib.libs.plotting.utils_latex_matplot import save_fig as save_fig_lib


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


# helper to allow us not to write the figs dir everywhere; todo: should be in a config...
figs_dir = Path("/Users/jsrozner/docs_local/research/proj_code/rozner-mono-cxs-main/proj/cxs_are_revealed/supplemental/figs")

def save_fig(
        fig: Figure,
        filename: str,
        transparent_bg = False
):
    """
    Will save a matplot Fig to filename

    Args:
        fig:
        filename:
        transparent_bg:

    Returns: None

    """
    save_fig_lib(fig, figs_dir, filename, transparent_bg)
