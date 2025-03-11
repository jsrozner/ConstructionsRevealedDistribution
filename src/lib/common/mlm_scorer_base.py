import hashlib

import torch
from memoization import cached
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, BertTokenizerFast, BertForMaskedLM

from rozlib.libs.library_ext_utils.utils_torch import get_device

allowed_non_roberta_models = [
    'smallbenchnlp/roberta-small',
    'phueb/BabyBERTa-1',
    'bert-large-uncased',
    '3van/RoBERTa_100M_ELI5_CurriculumMasking'
]

def _tensor_hasher(self, input_ids: torch.Tensor) -> str:
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

class MLMScorerBase:
    def __init__(
            self,
            model: str = 'roberta-large',
            output_attentions: bool = False,
    ):
        if not model.startswith('roberta') and not model in allowed_non_roberta_models:
            raise Exception(f"Invalid model {model} given - not roberta, not in {allowed_non_roberta_models}")

        self.device = get_device()

        if model.startswith('bert'):
            tokenizer_class = BertTokenizerFast
            model_class = BertForMaskedLM
        else:
            tokenizer_class = RobertaTokenizerFast
            model_class = RobertaForMaskedLM

        # fast tokenizer enables retrieiving char spans
        self.tokenizer: RobertaTokenizerFast = tokenizer_class.from_pretrained(
            model)
        # todo(low) - typing
        self.model: RobertaForMaskedLM = model_class.from_pretrained(
            model,
            output_attentions=output_attentions,
        )  # pyright: ignore [reportAttributeAccessIssue]
        self.model.to(self.device)  # pyright: ignore [reportArgumentType]
        self.model.eval()

    # todo: typing - outputs is likely a tuple or whatever torch is returning (see type error below)
    # todo(imp) limit cache size
    @cached(custom_key_maker=_tensor_hasher)
    def get_model_outputs_for_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Actually run the model forward.
        """
        with torch.no_grad():
            inputs = input_ids.to(self.device)
            outputs = self.model(inputs)

        return outputs

    def get_batch_encoding_for_sentence(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors='pt')
        return encoding
