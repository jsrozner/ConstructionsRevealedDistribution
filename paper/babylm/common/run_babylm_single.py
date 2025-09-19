"""
This is called via subcommand by run_exp.py

This boiler plate is necessary bc of how we set everything up with the separate file calls
We needed to be able to separately import the model and test list, unless we use reflections
    or better handle spawning processes, I think

All config is in run_babylm_top
"""

# todo: is this definitively importable?
try:
    # relative import (works when inside a package, ie when we run from cluster)
    from .fix_path import fix_path
except ImportError:
    # running locally via pycharm
    from babylm.common.fix_path import fix_path

def main():
    fix_path()

    # these could prob be imported above?, but we prob want fix path called first
    try:
        # relative import (works when inside a package, ie when we run from cluster)
        from .run_babylm_top import BabyLMConfig
        from .run_exp import single_run_log_config, single_run_main
    except ImportError:
        # running locally via pycharm
        from babylm.common.run_babylm_top import BabyLMConfig
        from babylm.common.run_exp import single_run_log_config, single_run_main

    model_list, test_list = BabyLMConfig.get_config()
    single_run_log_config()

    single_run_main(model_list, test_list)


if __name__ == "__main__":
    main()
