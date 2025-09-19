from __future__ import annotations
import hashlib
import logging
import os.path
import warnings
from typing import Optional

import torch
from memoization import cached
from transformers import BertTokenizerFast, BertForMaskedLM, PreTrainedTokenizerFast, PreTrainedModel, \
    AutoModelForMaskedLM, AutoTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from lib.common.mlm_models_config import all_allowed, bert_models, ltg_bert_models, model_mapping
from rozlib.libs.library_ext_utils.utils_torch import get_device
from rozlib.libs.utils.user_confirm import get_user_confirmation_with_record

_trust_hf_conf_save_file = "./trust_hf_remote_repo"

# requires signature to match, hence unused variable; see note below
def _tensor_hasher(mlm: MLMScorerBase, input_ids: torch.Tensor) -> str:
    """
    Custom key maker that hashes the tensor input for caching for use with memoized @cached.
    Function signature must match the signature of the cached function.
    """

    def hash_tensor(tensor: torch.Tensor) -> str:
        """Convert a tensor to a hashable string."""
        tensor_list = tensor.cpu().tolist()  # Convert tensor to list
        tensor_bytes = str(tensor_list).encode('utf-8')  # Convert to bytes
        return hashlib.sha256(tensor_bytes).hexdigest()  # Hash it

    # Hash the tensor input (ignoring `self`)
    hashed_input = hash_tensor(input_ids)

    return hashed_input  # Return unique cache key

# todo: should be abstract
class MLMScorerBase:
    def __init__(
            self,
            model: str = 'roberta-large',
            revision: Optional[str] = None,
            output_attentions: bool = False,
            use_cache = True
    ):
        logging.warning("INIT")
        self._model_name = model
        if not model.startswith('roberta') and not model in all_allowed :
            raise Exception(f"Invalid model {model} given - update allowed models in mlm_scorer_base.py")

        self.device = get_device()
        self.use_cache = use_cache
        if use_cache:
            logging.warning("Will use cache for MLMScorer. "
                            "Use this for analysis."
                            "For computation, avoid this; memory will blow up.")

        trust_remote = False
        if model in bert_models:
            tokenizer_class = BertTokenizerFast
            model_class = BertForMaskedLM
        else:
            tokenizer_class = AutoTokenizer
            model_class = AutoModelForMaskedLM

        if (model in ltg_bert_models or
                model.startswith('nikitas') or
                model.startswith("BabyLM-community")):
            user_conf = get_user_confirmation_with_record(
                _trust_hf_conf_save_file,
                "Ltg bert and nikitas models need to download from remote repo. "
                "Do you want to trust them and skip the warning message? (y/n) "
                "If no, you will be prompted by HF and can then review the repos."
                "If yes, won't reprompt for 7 days",
                max_age_days=7)
            if user_conf:
                warnings.warn(f"This repo has been set to autotrust remote; you prev gave confirmation which is stored "
                              f"in {os.path.abspath(_trust_hf_conf_save_file)}")
                trust_remote = True

        # fast tokenizer enables retrieiving char spans
        self.tokenizer: PreTrainedTokenizerFast = tokenizer_class.from_pretrained(
            model,
            use_fast=True,
            clean_up_tokenization_spaces = True
        )
        # print(self.tokenizer.init_kwargs)
        # assert self.tokenizer.clean_up_tokenization_spaces is False
        self.model: PreTrainedModel = model_class.from_pretrained(
            model,
            output_attentions=output_attentions,
            # default arg is "main" which is prob stupid and should have been None in HF repo?
            revision=revision if revision else "main",
            trust_remote_code=trust_remote
        )
        self.model.to(self.device)  # pyright: ignore [reportArgumentType]
        self.model.eval()

    ########
    # we have these couple methods so that we can cache or not cache outputs
    @staticmethod
    def __get_model_outputs_for_input(
            mlm: MLMScorerBase,
            input_ids: torch.Tensor,
            **kwargs
    ) -> MaskedLMOutput:
        with torch.no_grad():
            inputs = input_ids.to(mlm.device)
            outputs = mlm.model(inputs, return_dict=True, **kwargs)

        return outputs

    # @cached requires consistent type signature, but if it decorates a normal method, then external uses will complain that type sig does not match (since "self" is not passed)
    @staticmethod
    @cached(custom_key_maker=_tensor_hasher, ttl=300, max_size=3000)
    def _get_model_outputs_for_input_cached(
            mlm: MLMScorerBase,
            input_ids: torch.Tensor,
            **kwargs
    ) -> MaskedLMOutput:
        """
        Actually run the model forward.
        """
        return mlm.__get_model_outputs_for_input(mlm, input_ids, **kwargs)
    ########

    def get_model_outputs_for_input(
            self,
            input_ids: torch.Tensor,
            use_cache: Optional[bool] = None,
            **kwargs
    ) -> MaskedLMOutput:
        """
        Actually run the model forward.
        """
        # use_cache takes always overrides
        if use_cache is not None:
            do_use_cache = use_cache
        else:
            do_use_cache = self.use_cache

        if do_use_cache:
            return self._get_model_outputs_for_input_cached(self, input_ids, **kwargs)
        else:
            return self.__get_model_outputs_for_input(self, input_ids, **kwargs)

    def get_batch_encoding_for_sentence(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True)
        return encoding

    @property
    def allow_space_in_tokenization_span(self):
        if self._model_name not in model_mapping:
            return 0

        return model_mapping[self._model_name][0]

    @property
    def output_shift_for_decoding(self) -> int:
        # todo(fix this...maybe subclass)
        if self._model_name not in model_mapping:
            return 0

        return model_mapping[self._model_name][1]

    def maybe_shift_tok_idx_for_decoding(self, tok_idx: int) -> int:
        return tok_idx + self.output_shift_for_decoding

    # todo: might be better to just shift the ouputs when we run the model forward
    def extract_from_tensor_for_batch_and_tok_idx(self, logits: torch.Tensor, batch_idx, tok_idx: int):
        # check dims
        assert len(logits.shape) == 3   # batch, tokens, vocab
        return self.extract_from_tensor_for_tok_idx(logits[batch_idx], tok_idx)

    def extract_from_tensor_for_tok_idx(self, logits: torch.Tensor, tok_idx: int):
        assert len(logits.shape) == 2   # tokens, vocab
        # check dims
        actual_idx = self.maybe_shift_tok_idx_for_decoding(tok_idx)
        return logits[actual_idx]

