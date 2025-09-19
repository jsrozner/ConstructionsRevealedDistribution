"""
This file basically adapted from exp1/bnc_run_mlm_gpu.py

However we did not implement batching
"""
import os

from lib.exp_common.config import default_config
from lib.utils.utils_run import run_for_file
from proj.cxs_are_revealed.paper.cxns_in_distrib import run_for_leonie_with_surprisal
from rozlib.libs.common.config_wrapper import set_config
from rozlib.libs.common.logging.logging import setup_logging

# Suppress print globally
# import builtins
# builtins.print = lambda *args, **kwargs: None

def run_leonie_multithat():
    # local derived config
    input_dir = data_dir / "in_out_pairs/cec/multithat"
    output_dir = input_dir / "cluster_output"

    input_file = input_dir / "multithat_rozner_text_only.txt"

    run_for_file(
        input_file,
        output_dir,

        # todo: note original was with HHI; now we run with surprisal
        run_for_leonie_with_surprisal
    )


if __name__ == "__main__":
    print("Current Working Directory:", os.getcwd())

    # set up config
    _c = default_config
    print(_c)

    # work around to make sure logging inits... before setting c
    data_dir = _c.data_dir
    runs_log_dir = data_dir / "runs"
    setup_logging(runs_log_dir, write_print_to_log=False)

    # then we set config after logging is set up
    set_config(_c)

    run_leonie_multithat()
