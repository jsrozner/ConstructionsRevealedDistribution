"""
todo - some of these should go in rozlib
"""
import itertools
import random

from nltk import WordNetLemmatizer
from nltk.corpus import wordnet as wn, wordnet
from typing import List

from openai import ChatCompletion

from data_config import Exp4NPN
from lib.mlm_singleton import init_singleton_scorer
from paper.exp4_npn.npn_utils import VocabItem, GPTResult
from rozlib.libs.api.api_gpt import rungpt
from rozlib.libs.common.data.utils_jsonl import dump_to_jsonl

mlm_scorer = init_singleton_scorer('roberta-large', output_attentions=True)


def get_vocab() -> List[VocabItem]:
    """
    Get all vocab items in the tokenizer vocab
    """
    vocab: List[VocabItem] = []
    for k,v in mlm_scorer.tokenizer.vocab.items():
        vocab.append(VocabItem(
            v, k, mlm_scorer.tokenizer.convert_tokens_to_string([k])
        ))

    return sorted(vocab, key=lambda x: x.id)
    # v = mlm_scorer.tokenizer.vocab
    # return [mlm_scorer.tokenizer.convert_tokens_to_string([s]) for s in v.keys()]
    # return vocab

def get_word_pos(word: str) -> List[str]:
    """
    Retrieve all potential parts of speech for a given word using WordNet.

    Args:
        word (str): The word to lookup.
    """
    # pos_list = {
    #     'n': 'noun',
    #     'v': 'verb',
    #     'a': 'adjective',
    #     's': 'adjective (satellite)',
    #     'r': 'adverb',
    # }
    pos_types = ['n', 'v', 'a', 's', 'r']

    all_pos = [s.pos() for s in wn.synsets(word)]
    for p in all_pos:
        if not p in pos_types:
            print(f"invalid pos: {p}")

    return list(set(all_pos))

def is_valid_vocab_item(v: VocabItem,
                        valid_pos: List[str]):
    """
    Word must be
    - alphabetic
    - all lower case
    - start with a space (middle of word)
    - recognized / valid word
    - match particular word type using wordnet
    """
    if not v.str_rep.startswith(" "):
        # print("no space")
        return False
    w = v.str_rep.strip()
    if not w.isalpha():
        # print("not alpha")
        return False
    if not w.islower():
        # print("not lower")
        return False

    word_pos = get_word_pos(w)
    if len(word_pos) == 0:  # word did not exist?
        # print(f" {w} does not exist in wordnet")
        return False

    for valid in valid_pos:
        if valid in word_pos: return True

    # print(f"No pos match for {w} between {word_pos} and {valid_pos}")
    return False

def is_plural(word: str) -> bool:
    """
    Determine if a noun is plural using WordNet.

    Args:
        word (str): The input word.

    Returns:
        bool: True if the word is plural, False otherwise.
    """
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, pos="n")  # Lemmatize as a noun

    # Check if the lemmatized form is different and both exist in WordNet
    if lemma != word and wordnet.synsets(lemma, pos=wordnet.NOUN):
        return True  # The word is plural
    return False  # The word is singular or not found

def get_nouns():
    all_to_ret=[]
    vocab = get_vocab()
    for vocab_item in vocab:
        if not is_valid_vocab_item(vocab_item, ['n']):
            continue
        if is_plural(vocab_item.str_rep.strip()):
            # print(f"{vocab_item.str_rep} is plural")
            continue
        all_to_ret.append(vocab_item)

    return all_to_ret


def write_to_file(noun: str, prep: str, resp: ChatCompletion):
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
    dump_to_jsonl(gpt_result,Exp4NPN.npn_gpt_outputs)

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

def rungpt_with_prompt(noun: str, prep: str):
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

def run_gpt_and_write(noun: str, prep: str):
    res = rungpt_with_prompt(noun, prep)
    write_to_file(noun, prep, res)

def run():
    random.seed(42)
    all_nouns = get_nouns()
    sampled_nouns: List[VocabItem] = random.sample(all_nouns, 100)

    preps = ["to", "after", "by", "upon"]

    # for n, p in itertools.product(sampled_nouns, preps):
    #     print(n.str_rep_no_space,p)

    for n, p in itertools.product(sampled_nouns, preps):
        # todo: check that this is writing noun as string and not vocabitem
        run_gpt_and_write(n, p)


if __name__ == "__main__":
   run()