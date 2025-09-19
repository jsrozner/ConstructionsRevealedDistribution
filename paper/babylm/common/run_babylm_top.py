import pprint
import sys
from pathlib import Path
from typing import Callable


try:
    # relative import (works when inside a package, ie when we run from cluster)
    from .fix_path import fix_path
except ImportError:
    # running locally via pycharm
    from babylm.common.fix_path import fix_path

class BabyLMConfig:
    @staticmethod
    def get_config():
        fix_path()

        from babylm.exp1_cec import exp1_cec
        from babylm.exp2_multithat import exp2_multithat
        # todo: for some reason these are importing BabyLMModelList
        from babylm.exp3_analysis import exp3_magpie_process
        from babylm.exp3_magpie import exp3_magpie_jag
        from babylm.exp4_cogs import exp4_cogs
        from babylm.exp5_comp_corr import exp5_cc
        from babylm.exp6_npn_babylm import exp6_npn
        from babylm.exp_supp import count_params
        from babylm.config_babylm import BabyLMModelList

        model_list = BabyLMModelList()

        test_list: list[Callable] = [
            exp1_cec,
            exp2_multithat,

            # run remotely to produce data
            # exp3_magpie_jag,

            # run local for analysis
            # exp3_magpie_process,

            exp4_cogs,
            exp5_cc,
            exp6_npn,
            count_params,
        ]

        return model_list, test_list

def main():
    exec_path = fix_path()
    model_list, test_list = BabyLMConfig.get_config()

    from babylm.common.run_exp import run_all

    # will spawn a sub process for each model, test pair
    output_dir = Path("./output")
    run_all(
        output_dir,
        test_list,
        model_list,
        exec_path
    )


if __name__ == "__main__":
    print(f"running for {pprint.pformat(sys.argv)}")
    main()
