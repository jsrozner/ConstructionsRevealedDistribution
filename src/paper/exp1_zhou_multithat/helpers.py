from typing import List

from corpus_tools.zhou_cxs_so_difficult.corpus_leonie_eap_aap_cec import BaseExample


def str_with_indexes(input: str):
    """
    For adding labels to multithat examples - prints e.g.
    1 This 2 is 3 a 4 set 5 of 6 words
    """
    words = input.split(" ")
    to_print_list = [f"{i} {w}" for i,w in enumerate(words)]
    return " ".join(to_print_list)

def print_multithat_info(all_exs: List[BaseExample]):
    all_multithat = [e for e in all_exs if e.multi_that]
    for m in all_multithat:
        print(m.id)
        s = m.sentence_punct_fixed

        to_print = str_with_indexes(s)
        print(to_print)
