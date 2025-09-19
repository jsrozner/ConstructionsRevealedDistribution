import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path
from pprint import pp

from typing import List

from openai import ChatCompletion
from transformers import AutoTokenizer

from proj.cxs_are_revealed.paper.data_config import BabyLMExp6NPN, Exp4NPN
from lib.common.joint_vocab import get_vocab, VocabItem, compute_joint_vocab
from lib.utils.utils_run import check_dir_exists
from proj.cxs_are_revealed.paper.proj_common.npn_dataset_generation.npn_utils import GPTResult
from rozlib.libs.api.api_gpt import rungpt
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl
from rozlib.libs.library_ext_utils.utils_wordnet import get_word_pos, is_plural


# simple local run config
@dataclass
class NPNGenerationConfig:
    model: str
    # todo: tokenization schemes vary; this is not being used
    #   todo - make note of where we are looking at this in joint_vocab.compute_joint_vocab
    model_require_preceding_space: bool
    min_word_len: int
    gpt_output_file: Path
    num_generations: int = 100
    # note that seeding will not produce the same outputs we generated since our nouns list was not sorted when we generated
    seed: int = 42
    do_generate: bool = False
    compute_joint_vocab: bool = False
    filter_likely_gerunds: bool = False


# for original run original exp
config_original = NPNGenerationConfig(
    model = 'roberta-large',
    model_require_preceding_space=True,
    gpt_output_file= Exp4NPN.npn_gpt_outputs,
    min_word_len=0,
    do_generate=False

)

# resample for babyLM
config_babyLM = NPNGenerationConfig(
    model = 'ltg/gpt-bert-babylm-small',
    model_require_preceding_space=False,
    min_word_len=4,
    gpt_output_file= BabyLMExp6NPN.npn_gpt_outputs,
    compute_joint_vocab = True,
    filter_likely_gerunds=True,

    # set true to actually run
    # do_generate=True,

    num_generations=4,
    seed = 90,
)

def is_valid_for_experiment(
        word: str,
        min_length = 4,
        filter_likely_gerunds=True
):
    """
    Word must be
    - alphabetic
    - all lower case
    - start with a space (middle of word)
    - recognized / valid word
    - match particular word type using wordnet
    """
    if not word.isalpha():
        return False
    if not word.islower():
        return False

    word_pos = get_word_pos(word, case_sensitive=True)
    if len(word_pos) == 0:  # word did not exist?
        # print(f" {w} does not exist in wordnet")
        return False

    if 'n' not in word_pos:
        return False

    if filter_likely_gerunds and word.endswith("ing"):
        return False
    if is_plural(word):
        return False
    if len(word) < min_length:
        return False

    return True


def write_to_file(
        noun: str,
        prep: str,
        resp: ChatCompletion,
        file: Path
):
    if not len(resp.choices) == 1:
        print(f"more than one response")
        print(resp)
    gpt_result = GPTResult(
        noun,
        prep,
        resp.model,
        resp.choices[0].message.content,
        resp.choices[0].finish_reason
    )
    dump_to_jsonl(gpt_result, file)

"""
Sample words, write to file
- return string
- return chat completion subpart
- dump jsonl

check
- has only one period
- starts with capital letter
- contains target string (needs to all be lower case)
- check whether the word is repeated (interesting case)

other
- make note of pricing

subseq experiments
- examine what substitution of the other words for the sentence does (other top fills if we double mask)
"""

def rungpt_with_prompt(noun: str, prep: str) -> ChatCompletion:
    phrase = f"{noun} {prep} {noun}"

    resp = rungpt(
        model="gpt-4",
        content= f'An NPN construction is one like "day by day" or "face to face". It has a repeated singular noun '
                        f'with a preposition in the middle. '
                        f'Other prepositions are also possible: "book upon book", "week over week", "year after year". '
                        f'Please use "{phrase}" in an NPN construction, '
                        f'placing "{phrase}" in the middle of the sentence. '
                        'Make sure the sentence establishes a context in which the noun makes sense. '
                        'Please provide only the sentence in the response.',
        user="research-npn",
        temperature=0.7,
        max_tokens=100,
    )
    return resp

def run_gpt_and_write(
        noun: str,
        prep: str,
        file: Path
):
    res = rungpt_with_prompt(noun, prep)
    write_to_file(noun, prep, res, file)

def run():
    random.seed(config.seed)
    tok = AutoTokenizer.from_pretrained(config.model, use_fast=True, clean_up_tokenization_spaces=True)
    all_vocab: List[VocabItem] = get_vocab(tok)

    if config.compute_joint_vocab:
        # hack: this import ideally would not be here (imports in the wrong direction, from another project)
        from proj.cxs_are_revealed.exp import model_list

        print("filtering to joint vocab")
        all_models = list(model_list.keys())
        joint_vocab = compute_joint_vocab(all_models)
        print(f"len of joint vocab is {len(joint_vocab)}")
        all_vocab = [x for x in all_vocab if x.str_rep_no_space in joint_vocab]
    # print(len(all_vocab))
    # todo note that we probably have words with/without preceding space
    voc = set([x.str_rep_no_space for x in all_vocab])
    # print(len(set(voc)))

    all_nouns = [
        x for x in voc
        if is_valid_for_experiment(
            x,config.min_word_len, config.filter_likely_gerunds)
    ]
    all_nouns = sorted(all_nouns)
    print(f"got possible nouns: {len(all_nouns)}")

    # if should compute joint vocab, compute

    sampled_nouns: List[str] = random.sample(all_nouns, config.num_generations)
    # nouns = [x.str_rep_no_space for x in sampled_nouns]
    pp(sorted(sampled_nouns))

    preps = ["to", "after", "by", "upon"]

    if config.do_generate:
        path = config.gpt_output_file
        check_dir_exists(os.path.dirname(path), create_ok=True)

        for n, p in itertools.product(sampled_nouns, preps):
            run_gpt_and_write(n, p, config.gpt_output_file)
    else:
        print("do_generate not set; will not call GPT api")


if __name__ == "__main__":
    # config = config_babyLM
    config = config_original
    pp(config)
    run()
