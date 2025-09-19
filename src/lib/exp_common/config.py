from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from proj.cxs_are_revealed.paper.data_config import DATADIR
# todo: this should be importing from this project (we should use the config_wrapper but in another local file, I think)
from rozlib.libs.common.config_wrapper import _config, Config

# Write a new config function that returns the correct typing
def get_config() -> MLMConfig:
    return _config()  # pyright: ignore [reportReturnType]

"""
Configuration details, e.g. for local comp or cluster
"""
@dataclass
class CompConfig:
    proj_root: Path

@dataclass
class MLMConfig(Config):
    data_dir: Path
    float_decimals: int = 4

""" Computer configs """
comp_config_mac = CompConfig(
    proj_root = Path("/Users/jsrozner/docs_local/research/projects/research_constructions/constructions_repo/")
)

# todo: cluster config

john13_config = MLMConfig(
    data_dir=Path("/john13/scr1/biggest/enroot/roz_constructions/data")
)

# also for jag38
jag39_config = MLMConfig(
    data_dir=Path("/scr/rozner/data")
)

jag30_config = MLMConfig(
    data_dir=Path("/jagupard30/scr1/rozner/data")
)
jag19_config = MLMConfig(
    data_dir=Path("/jagupard19/scr0/rozner")
)

# for distribution in repo
default_config = MLMConfig(
    data_dir=DATADIR
)

def get_default_config() -> MLMConfig:
    return default_config


def make_config(cc: CompConfig):
    return MLMConfig(
        data_dir = cc.proj_root / "data",
    )

def make_config_mac():
    return make_config(comp_config_mac)
