"""
Produces corpus_leonie_all_processed.txt

This will output all 5 sentence types for the Zhou paper.
We do not actually need all 5 sentence types.
"""
from pathlib import Path
from typing import List

from corpus_tools.zhou_cxs_so_difficult.corpus_leonie_eap_aap_cec import get_clean_exs, BaseExample
from proj.cxs_are_revealed.paper.data_config import Exp1Zhou


def write_to_file(
        tgt_path: Path,
        exs: List[BaseExample],
        include_all = True
):
    with open(tgt_path, "w") as f:
        for ex in exs:
            f.write(ex.o + "\n")
            if include_all:
                f.write(ex.ds + "\n")
                f.write(ex.dt + "\n")
                f.write(ex.dst + "\n")
                f.write(ex.an + "\n")

if __name__ == "__main__":
    exs_no_errors = get_clean_exs(Exp1Zhou.zhou_original_xlsx)

    raise Exception(f"if you would like to regenerate the data, uncomment")

    # original data includes all 5 perturbations described in zhou paper (but we never use the non-original sentence)
    write_to_file(Exp1Zhou.zhou_preprocessed_all_sents, exs_no_errors, include_all = False)

    # produces only the originals (re-produced later, for camera ready version)
    write_to_file(Exp1Zhou.zhou_preprocessed_core, exs_no_errors, include_all = False)
