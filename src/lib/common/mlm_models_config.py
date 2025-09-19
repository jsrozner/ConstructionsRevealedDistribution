# Model mapping indcates whether to
# - allow space in tokenization (bool)
# - amount of output shift
# todo: note that we also have variability in whether the token has a preceding character for spacing (the funny G or an _)
from typing import Dict, Tuple

model_mapping: Dict[str, Tuple[bool, int]] = {
    'ltg/gpt-bert-babylm-small': (True, -1),
    'ltg/gpt-bert-babylm-base': (True, -1),

    'babylm/ltgbert-100m-2024': (True, 0),
    'babylm/ltgbert-10m-2024': (True, 0),

    # also an ltg architecture
    'nikitastheo/BERTtime-Stories-100m-nucleus-1': (True, 0),
    'nikitastheo/BERTtime-Stories-10m-nucleus-1-balanced': (True, 0),

    'jdebene/BabyLM2024': (True, 0),

    # todo for acquisition study
    "BabyLM-community/babylm-baseline-100m-gpt-bert-mixed": (True, -1),
    "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus": (True, -1),
    "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus": (True, -1),
}

#nonroberta means not roberta-base or roberta-large
allowed_non_roberta_models = [
    '3van/RoBERTa_100M_ELI5_CurriculumMasking',
    '3van/RoBERTa_10M_BabyLMBaseline_CurriculumMasking',
]

bert_models = [
    'google-bert/bert-base-cased',
    'google-bert/bert-large-cased',
]

ltg_bert_models = [
    # ltg models are not bert models; they have special config
    'ltg/gpt-bert-babylm-small',
    'ltg/gpt-bert-babylm-base',
    'ltg/ltg-bert-babylm',
    'babylm/ltgbert-100m-2024',
    'babylm/ltgbert-10m-2024',
    'nikitastheo/BERTtime-Stories-100m-nucleus-1',
]

all_allowed = allowed_non_roberta_models + bert_models + ltg_bert_models + list(model_mapping.keys())

# notes
# small bert (both do not work)
# 'lyeonii/bert-small',       # does not work
# 'smallbenchnlp/bert-small', # does not work

# small roberta models
# 'smallbenchnlp/roberta-small',
# 'phueb/BabyBERTa-1',

# deberta models that aren't working; maybe a tokenizer issue
# "SzegedAI/babylm24_MLSM_strict": (True, 0),
# "SzegedAI/babylm24_LSM015_strict": (True, 0)

# '3van/RoBERTa_100M_ELI5_Baseline',
# roberta model; gives an error..deprecated config. not sure
