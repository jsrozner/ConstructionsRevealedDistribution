"""
Run exp1 on cluster.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from data_config import Exp1Zhou
from lib.exp_common.config import jag39_config, default_config
from lib.scoring_fns import surprisal, probability
from rozlib.libs.common.config_wrapper import set_config
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl
from rozlib.libs.common.logging.logging import setup_logging
from lib.distr_diff_fcns import jensen_shannon_divergence
from lib.exp_common.mlm_gpu_affinity import process_sent_affinity
from rozlib.libs.library_ext_utils.utils_torch import clear_gpu_cache


def run_for_leonie_with_surprisal(
        input_file: Path,
        output_file: Path,
        line_limit: Optional[int] = None,
        skip_ct = None
):
    """
    For each line in input_file, process and write to output
    """
    logging.info(f"input: {input_file}; output: {output_file}")
    file_id = input_file.name.split(".")[0]
    with open(input_file, 'r') as f:
        line_ct = 0
        for idx, line in tqdm(enumerate(f)):
            if line_limit is not None and line_ct >= line_limit:
                break
            line_ct += 1
            if skip_ct is not None and line_ct <= skip_ct:
                continue
            try:
                result = process_sent_affinity(
                    file_id=file_id,
                    sent_idx_in_file=idx,
                    sent=line.strip(),  # remove newline
                    # score_fn=surprisal,
                    score_fn=probability,
                    calculate_affinities=True,
                    dist_dff_fn=jensen_shannon_divergence
                )
                # write
                dump_to_jsonl(result, output_file)
            except Exception as e:
                logging.error(f"While processing {file_id}, {idx}\n\t{line}\n "
                              f"Exception {e}")
            clear_gpu_cache()

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

    # this was fixed for emnlp camera ready (redo eap/aap local affinity with jsd)
    run_for_leonie_with_surprisal(
        Exp1Zhou.zhou_preprocessed_core,
        Exp1Zhou.zhou_all_core_outputs,
        skip_ct=14
    )
