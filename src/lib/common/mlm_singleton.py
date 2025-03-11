"""
Use this module when we want to share a single mlm_scorer across multiple modules.
Enables easy transition from jupyter notebooks to putting code in python files.
"""
import logging

from lib.mlm_scorer import MLMScorer

_mlm_scorer_singleton = None


def init_singleton_scorer(model='roberta-large', output_attentions=False):
    global _mlm_scorer_singleton
    if _mlm_scorer_singleton is not None:
        print("WARN: singleton already initialized; will not re-init")
        return _mlm_scorer_singleton
    logging.warning(f"Initializing {model}")
    _mlm_scorer_singleton = MLMScorer(model, output_attentions=output_attentions)
    return _mlm_scorer_singleton


def get_singleton_scorer(model='roberta-large', allow_init=True) -> MLMScorer:
    if not allow_init and _mlm_scorer_singleton is None:
        raise Exception("call init_scorer first")
    if _mlm_scorer_singleton is None:
        init_singleton_scorer(model)
    return _mlm_scorer_singleton  # pyright: ignore [reportReturnType] todo

