"""
Use this module when we want to share a single mlm_scorer across multiple modules.
Enables easy transition from jupyter notebooks to putting code in python files.
"""
import logging
import warnings
from typing import Optional

from lib.mlm_scorer import MLMScorer
from rozlib.libs.library_ext_utils.utils_torch import clear_gpu_cache

_mlm_scorer_singleton: Optional[MLMScorer] = None
_model = None
default_model = 'roberta-large'

# todo: we should use a true module singleton with metaclass
def init_singleton_scorer(
        model: Optional[str] = None,
        revision: Optional[str] = None,
        output_attentions=False,
        use_cache = False
):
    if model is None:
        # todo: we should not allow this for safety
        logging.warning(f"Init singleton called with no model; will use {default_model}")
        model = default_model

    global _mlm_scorer_singleton, _model

    # We allow init to be called and return existing - check if we already initialized
    if _mlm_scorer_singleton is not None:
        logging.warning(f"WARN: singleton already initialized ({_model}); "
                        f"will not re-init")
        if _model != model:
            raise Exception(f"Models do not match {model} {_model}")
        if not use_cache and _mlm_scorer_singleton.use_cache:
            raise Exception("requested no cache but cache is set")
        return _mlm_scorer_singleton

    # actually init
    logging.warning(f"Initializing {model}, {revision}")
    _model = model
    _mlm_scorer_singleton = MLMScorer(
        model,
        output_attentions=output_attentions,
        revision=revision,
        use_cache=use_cache
    )
    # wanted to see where inits were being called from
    # traceback.print_stack()
    return _mlm_scorer_singleton

def set_singleton_scorer(
        mlm: MLMScorer,
        # model: str
):
    raise NotImplemented
    # # todo: we probably want to make sure that anyone that had a prev version...knows that it changed
    # global _mlm_scorer_singleton, _model
    # # if mlm._model_name != _model:
    # #     raise Exception("model names do not match")
    # _mlm_scorer_singleton = mlm
    # _model = mlm._model_name


def get_singleton_scorer(
        # model='roberta-large',
        model=None,
        allow_init=True,
        use_cache: bool = False
) -> MLMScorer:
    if _mlm_scorer_singleton is not None:
        if use_cache is False and _mlm_scorer_singleton.use_cache:
            raise Exception("requested no cache; but cache is set")
        if model is not None:
            assert _mlm_scorer_singleton._model_name == model
        return _mlm_scorer_singleton

    # the module singleton is None
    if not allow_init:
        raise Exception("call init_scorer first")
    # todo: for safety allow_init should be false?
    logging.warning(f"Get_singleton called but no model is initialized; will call init")
    return init_singleton_scorer(model, use_cache=use_cache)


# todo: not sure when / if this is used; maybe remove
def reset_singleton_scorer():
    warnings.warn("you should verify that this function does what you want it to do")
    global _mlm_scorer_singleton, _model
    if _mlm_scorer_singleton is None:
        logging.warning("called reset on mlm but it was not prev initialized")
        return

    # probably does nothing
    del _mlm_scorer_singleton.model
    del _mlm_scorer_singleton

    _mlm_scorer_singleton = None
    _model = None

    clear_gpu_cache()
