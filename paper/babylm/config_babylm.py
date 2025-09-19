from abc import ABC, abstractmethod
from typing import Dict, Tuple

from lib.common.mlm_scorer_base import MLMScorerBase
from lib.common.mlm_singleton import init_singleton_scorer

class ModelList(ABC):
    @classmethod
    @abstractmethod
    def get_model_iterator(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def model_shortname(cls, model_name: str):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_model_by_idx(cls, idx: int) -> tuple[str, MLMScorerBase]:
        raise NotImplementedError


class BabyLMModelList(ModelList):
    model_list: Dict[str, str] = {
        "ltg/gpt-bert-babylm-base": "gpt-bert-base",
        "babylm/ltgbert-100m-2024": "ltgbert-baseline-100m",
        "nikitastheo/BERTtime-Stories-100m-nucleus-1": "berttime-100m",
        "3van/RoBERTa_100M_ELI5_CurriculumMasking": "3van-100m",

        "ltg/gpt-bert-babylm-small": "gpt-bert-small",
        "nikitastheo/BERTtime-Stories-10m-nucleus-1-balanced": "berttime-10m",
        "jdebene/BabyLM2024": "jdebene-10m",
        "babylm/ltgbert-10m-2024": "ltgbert-baseline-10m",

        'roberta-large': "rob-large",
        'roberta-base': "rob-base",
        'google-bert/bert-large-cased': "bert-large",
        'google-bert/bert-base-cased': "bert-base",

        # note that this one gives an error
        # "3van/RoBERTa_10M_BabyLMBaseline_CurriculumMasking": "3van-10m",
    }

    reverse_model_dict = {v: k for k, v in model_list.items()}

    @classmethod
    def get_model_iterator(cls):
        for m in cls.model_list:
            yield m

    @classmethod
    def model_shortname(cls, model_name: str):
        return cls.model_list[model_name]

    @classmethod
    def get_model_by_idx(cls, idx: int) -> tuple[str, MLMScorerBase]:
        model_name = list(cls.model_list.keys())[idx]
        return model_name, init_singleton_scorer(model_name)