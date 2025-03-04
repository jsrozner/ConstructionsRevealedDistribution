"""
Run exp1 on cluster.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from lib.exp_common.config import jag39_config, default_config
from lib.scoring_fns import surprisal
from lib.utils.utils_run import run_for_sub_dir
from rozlib.libs.common.config_wrapper import set_config
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl
from rozlib.libs.common.logging.logging import setup_logging
from lib.distr_diff_fcns import jensen_shannon_divergence
from lib.exp_common.mlm_gpu_affinity import process_sent_affinity


def run_for_leonie_with_surprisal(
        input_file: Path,
        output_file: Path,
        line_limit: Optional[int] = None,
):
    """
    For each line in input_file, process and write to output
    """
    file_id = input_file.name.split(".")[0]
    with open(input_file, 'r') as f:
        line_ct = 0
        for idx, line in tqdm(enumerate(f)):
            if line_limit is not None and line_ct >= line_limit:
                break
            try:
                result = process_sent_affinity(
                    file_id=file_id,
                    sent_idx_in_file=idx,
                    sent=line.strip(),  # remove newline
                    score_fn=surprisal,
                    calculate_affinities=True,
                    dist_dff_fn=jensen_shannon_divergence

                )
                # write
                dump_to_jsonl(result, output_file)
            except Exception as e:
                logging.error(f"While processing {file_id}, {idx}\n\t{line}\n "
                              f"Exception {e}")
            # todo: torch seems to have memory leaks; these are not working locally
            # gc.collect()
            # torch.cuda.empty_cache()

def run_leonie():
    # local derived config
    input_dir = data_dir / "corpus_parsed"
    output_dir = data_dir / "output/leonie"

    # run on sub dir
    run_for_sub_dir(
        input_dir,
        output_dir,
        # we are adapting previous code to work with single dir
        # ["leonie"],
        ["exp5_leonie_cec"],

        # todo: note original was with HHI; now we run with surprisal
        # run_for_preprocessed_bnc_sentence_file,
        run_for_leonie_with_surprisal,
        # keep_existing=True
        # file_limit_ct=1
    )

if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())

    _c = default_config
    print(_c)

    # work around to make sure logging inits... before setting c
    data_dir = _c.data_dir
    runs_log_dir = data_dir / "runs"
    setup_logging(runs_log_dir, write_print_to_log=False)

    # then we set config after logging is set up
    set_config(_c)

    run_leonie()
