import hashlib
from collections import Counter
from typing import Optional, List, Tuple

import torch
from memoization import cached
from transformers import RobertaForMaskedLM, RobertaTokenizerFast, BertTokenizerFast, BertForMaskedLM

from lib.exp_common.mlm_align_tokens import reassembled_words_from_tokens_roberta, TokenizedWordInSentence, \
    align_words_with_token_list
from lib.scoring_fns import hhi_rounded, entropy_rounded, ScoreFn, hhi_trunc_rounded, hhi_ratio_unrounded, step_score, prob_ratio
from lib.utils.utils_misc import get_nth_occ

from lib.scoring_fns import step_score, entropy_rounded, hhi_rounded, prob_ratio
from rozlib.libs.utils.string import split_and_remove_punct
from rozlib.libs.library_ext_utils.utils_torch import round_tensor, get_device

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


class MLMScorer:
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
    # todo limit cache size
    @cached(custom_key_maker=_tensor_hasher)
    def get_model_outputs_for_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Actually run the model forward.
        """
        with torch.no_grad():
            inputs = input_ids.to(self.device)
            outputs = self.model(inputs)

        return outputs

    # todo: deprecate this function
    def get_logits_for_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Actually run the model forward.
        """
        outputs = self.get_model_outputs_for_input(input_ids)
        return outputs.logits

    def get_batch_encoding_for_sentence(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors='pt')
        return encoding

    def prepare_inputs_for_sentence(self, sentence: str):
        encoding = self.tokenizer(sentence, return_tensors='pt')

        input_ids = encoding['input_ids']

        # tokens = encoding.tokens()[0]
        tokens = encoding.tokens()

        return input_ids, tokens

    # todo: remove
    def _prepare_inputs_for_sentence(self, sentence: str):
        input_ids: List[List[int]] = self.tokenizer.encode(sentence, return_tensors='pt')  # pyright: ignore [reportUnknownMemberType]
        tokens: List[str] = self.tokenizer.convert_ids_to_tokens(input_ids[0])  # get 1st sent in batch

        return input_ids, tokens

    def get_attns_for_sentence(self, sentence: str):
        input_ids, tokens = self.prepare_inputs_for_sentence(sentence)
        outputs = self.get_model_outputs_for_input(input_ids)

        return outputs.attentions, tokens

    def get_logits_for_word_in_sentence_by_idx(
            self,
            sentence: str,
            word_idx: int,
    ) -> Optional[torch.Tensor]:
        """
        Original method, NO BATCHING!; NOT OPTIMIZED
        # todo: this should use SentenceForMLMProcessing; it duplicates SentenceForMLMProcessing._prepare_sentences
        # (we rewrote the other for batch processing vs this one does not do batch processing)

        Returns the logits for the word at word_idx in sentence

        Returns
        - tensor of logits
        - None if word is multiply tokenized (and warns)
        """

        # todo(nc): why did we need to use old method
        # tokenize
        input_ids, tokens = self._prepare_inputs_for_sentence(sentence)
        tokenized_words = reassembled_words_from_tokens_roberta(tokens)

        # align words with tokenization
        sent_words_list = split_and_remove_punct(sentence)
        aligned: List[TokenizedWordInSentence] = align_words_with_token_list(sent_words_list, tokenized_words)
        tgt_word = sent_words_list[word_idx]
        aligned_token_word = aligned[word_idx]

        # todo: why are we checking this here
        if tgt_word == "mask":
            assert aligned_token_word.str_rep_no_special == "<mask>"
        else:
            assert aligned_token_word.str_rep_no_special == tgt_word, f"{aligned_token_word.str_rep_no_special} != {tgt_word}"

        # warn if it's multi tokenized
        if len(aligned_token_word.tokens) > 1:
            print(f"WARN: {sent_words_list[word_idx]} mult tokens {aligned_token_word.tokens}")
            return None
        elif len(aligned_token_word.tokens) != 1:
            raise Exception("unexpected; token length is 0?")

        token_idx: int = aligned_token_word.tok_idx_start

        # Mask the current token (replace with [MASK])
        # todo: what is wrong with typing here
        masked: int = input_ids[0, token_idx]
        masked_word = self.tokenizer.decode([masked])  # pyright: ignore [reportUnknownMemberType]
        if masked_word.strip() != tgt_word:
            # todo: without strip we have an issue because spaces treated diff?
            print(f"WARN: masked_word {masked_word}, not {tgt_word}")
        input_ids[0, token_idx] = self.tokenizer.mask_token_id

        # get likely fills
        outputs = self.get_model_outputs_for_input(input_ids)
        logits = outputs.logits

        # predictions is vocab_len [logit, logit, ... logit]
        token_logits = logits[0, token_idx]     # batch, idx, then vocab_len shape

        return token_logits

    # todo: copied from aboe function
    def get_logits_for_word_as_double_mask(
            self,
            sentence: str,
    ) -> Tuple[torch.Tensor]:
        """
        """
        # sent_words_list = split_and_remove_punct(sentence)
        # print(sent_words_list)
        # mask2_idx = sent_words_list.index("<mask2>")

        # tokenize
        input_ids, tokens = self.prepare_inputs_for_sentence(sentence)
        print(tokens)
        print(input_ids)

        if tokens.count("<mask>") != 2:
            raise ValueError()

        mask1_idx = tokens.index("<mask>")
        assert tokens[mask1_idx] == tokens[mask1_idx + 1] == "<mask>"

        # get likely fills
        outputs = self.get_model_outputs_for_input(input_ids)
        logits = outputs.logits

        # predictions is vocab_len [logit, logit, ... logit]
        token_logits = logits[0, mask1_idx:mask1_idx+2]     # batch, idx,

        return input_ids, token_logits

    def get_logits_for_masks_in_sentence(
            self,
            sentence: str,
    ) -> List[torch.Tensor]:
        sent_words_list = split_and_remove_punct(sentence)
        all_logits = []
        for idx, w in enumerate(sent_words_list):
            if w != "<mask>":
                continue
            logits = self.get_logits_for_word_in_sentence_by_idx(
                sentence,
                idx
            )
            all_logits.append(logits)

        return all_logits

    def get_logits_for_word_in_sentence(
            self,
            sentence: str,
            word: str,
            word_occ: int | None = None,
    ) -> Optional[torch.Tensor]:
        """
        Returns logits for all vocab for word in sentence

        word_occ - if multiple occurrences of the word, provide the index of the one you want (0 indexed)
        """
        sent_words_list = split_and_remove_punct(sentence)

        word_occurrence_ct = sent_words_list.count(word)
        if word_occurrence_ct > 1:
            if word_occ is None:
                raise Exception(f"{word} occurs more than once in sentence; specify which occurrence you want with word_occ (0 indexed)")
            # word_occ is 0 indexed here but 1 indexed in get_nth_occ
            word_idx = get_nth_occ(sent_words_list, word, word_occ + 1)
        else:
            word_idx = sent_words_list.index(word)

        return self.get_logits_for_word_in_sentence_by_idx(
            sentence,
            word_idx,
        )

    def _print_preds(self, logits: torch.Tensor | None, num_to_print: int = 5):
        """
        Given logits, will print
        - entropy, hhi, step_score for the logits as well as
        - num_to_print top predictions for the position based on the logits
        """
        if logits is None:    # sometimes can be None
            return

        # todo: enable to pass a list of score functions
        print(f"Entropy: {entropy_rounded(logits)}; HHI: {hhi_rounded(logits)}; step_score: "
              f"{step_score(logits)} ")

        if num_to_print <= 0:
            return

        predicted_ids = torch.topk(logits, num_to_print).indices
        token_logits = logits[predicted_ids]

        probs = prob_ratio(token_logits)
        for idx, id in enumerate(predicted_ids):
            print(f"{self.tokenizer.decode([id])} - {round(token_logits[idx].item(), 2)}"
                  f" - "
                  f"{round(probs[idx].item(), 3)}")
        # print("-" * 40)

    def print_preds(self, sent: str, word: str, num_to_print:int = 3):
        """
        For word in sent, calls _print_preds()
        Note that if word occurs multiple times it will warn and return the first one
        """
        self._print_preds(self.get_logits_for_word_in_sentence(sent, word), num_to_print)

    def get_logits_for_all_words_in_sentence(self, sent: str):
        """
        Yields
        - idx of word in sent (int)
        - w - the word (str)
        - logits: torch.Tensor
        """
        word_counter: Counter[str] = Counter()
        for idx, w in enumerate(split_and_remove_punct(sent)):
            logits = self.get_logits_for_word_in_sentence(sent, w, word_counter[w])
            word_counter[w] += 1
            yield idx, w, logits

    def print_preds_all(self, sent: str, num_to_print:int = 3):
        """
        For each word in sent, will call print_preds()
        """
        for idx, w, logits in self.get_logits_for_all_words_in_sentence(sent):
            print(w)
            if logits is not None:
                self._print_preds(logits, num_to_print=num_to_print)
            else:
                print("No logits (probably multitokenized?")
            print("-" * 40)

    def get_topk_preds_for_logits(
            self, logits: torch.Tensor, top_k: int = 10
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Return a list of the top predictions as tuples [word, score]
        """
        predicted_vocab_ids = torch.topk(logits, top_k).indices
        token_logits = logits[predicted_vocab_ids]
        probs = prob_ratio(token_logits)
        # words = tokenizer.batch_decode([predicted_vocab_ids])
        # print(words)
        words = [self.tokenizer.decode([id]) for id in predicted_vocab_ids]
        # print(words)
        return list(zip(words, probs))

    def _get_scores_for_logits(
            self,
            score_fns: List[ScoreFn],
            logits: torch.Tensor
    ) -> List[float] | None:
        """
        Compute a list of scores on logits using score_fns
        """
        scores: List[float] = []
        try:
            logits_bk = logits.clone()
        except:
            return None
        for score_fn in score_fns:
            if not torch.equal(logits, logits_bk):
                raise Exception("logits changed")
            # note that logits could be None #todo - does it need to be some other value?
            scores.append(round(score_fn(logits), 3))
        return scores

    def score_and_preds(
            self,
            score_fns: List[ScoreFn],
            sent: str,
            idx: int,
            top_k: int =10
    ):
        """
        For sentence, sent, and word index, idx, returns a tuple of
        - List[score_fn(logits_for_word_at_index)]
        - Top_k predictions for the word at idx
        """
        logits = self.get_logits_for_word_in_sentence_by_idx(sent, idx)

        if logits is None:
            return None, None

        scores = self._get_scores_for_logits(score_fns, logits)
        preds = self.get_topk_preds_for_logits(logits, top_k)
        pred_words = [p[0] for p in preds]
        pred_scores = [round_tensor(p[1]) for p in preds]
        preds = list(zip(pred_words, pred_scores))
        return scores, preds

    def _top_scores(self, sent: str, num_to_return: int =3, score_fn: ScoreFn = hhi_rounded, min_score_fn_score: float =.92):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        scores: List[Tuple[str, float]] = []
        for idx, w in enumerate(split_and_remove_punct(sent)):
            logits = self.get_logits_for_word_in_sentence_by_idx(sent, idx)
            if logits is None: continue  # e.g. for punctuation
            scores.append((w, score_fn(logits)))

        return sorted(
            # first filter
            filter(lambda x: x[1] > min_score_fn_score, scores),
            # then sort by and return only num_to_return
            key = lambda x: x[1], reverse=True)[:num_to_return]

    def top_scores_step(self, sent: str, num_to_return=3):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        return self._top_scores(sent, num_to_return, step_score )

    def top_scores_hhi(self, sent: str, num_to_return=3, min_hhi=0.92):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        return self._top_scores(sent, num_to_return, hhi_rounded, min_hhi)

    def top_scores_hhi_trunc(self, sent: str, num_to_return=3, min_score=0.92):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        return self._top_scores(sent, num_to_return, hhi_trunc_rounded, min_score)

    def top_scores_hhi_ratio(self, sent: str, num_to_return=3, min_score=0.92):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        return self._top_scores(sent, num_to_return, hhi_ratio_unrounded, min_score)

    def top_scores_entropy(self, sent: str, num_to_return=3, min_score=0.92):
        """
        Return the num_to_return positions in the sentence with the highest hhi score, sorted
        """
        return self._top_scores(sent, num_to_return, entropy_rounded, min_score)
