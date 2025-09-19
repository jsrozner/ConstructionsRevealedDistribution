import string

import spacy

from proj.cxs_are_revealed.paper.cxns_in_distrib.exp2_cogs.cogs_utils import CogsEntry

nlp_sm = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_trf")

def trim_punctuation(text: str) -> str:
    return text.strip(string.punctuation)

def is_comparative(word: str) -> bool:
    doc = nlp_sm(word)  # Process the word using SpaCy's NLP pipeline
    # print([token.tag_ for token in doc])
    return any(token.tag_ == "JJR" or token.tag_ == "RBR" for token in doc)  # Check if any token is a comparative adjective

# use transformer to get better context
def is_comparative_with_context(
        sentence: str,
        char_offset: tuple[int, int],
        use_tf = True
) -> bool:
    """
    Checks if the word at the given character offsets in the sentence is a comparative adjective or adverb.

    Args:
        sentence (str): The input sentence.
        char_offset (tuple[int, int]): The (start, end) character indices for the word.

    Returns:
        bool: True if the word is a comparative adjective (JJR) or adverb (RBR), False otherwise.
    """
    # fast return if we can do with single word
    if is_comparative(sentence[char_offset[0]:char_offset[1]]):
        return True

    if use_tf:
        doc = nlp(sentence)  # Process the full sentence
    else:
        doc = nlp_sm(sentence)
    for token in doc:
        if token.idx == char_offset[0] and token.idx + len(token.text) == char_offset[1]:
            return token.tag_ in {"JJR", "RBR"}  # Check for comparative adjective/adverb

    # todo: will print too much in babyLM experiment
    # assert False
    # if we didn't find
    print(f"in {sentence}, looking for ({sentence[char_offset[0]:char_offset[1]]})")
    # for t in doc:
    #     print(t)
    #     print(t.idx, t.idx + len(t.text))
    print("token not found!!")

    return False  # If no matching token is found

def check_cc(cc: CogsEntry):
    sent_words = cc.sent.split(" ")
    assert len(cc.tgt_words) == 2
    for i, sent_word_idx in enumerate(cc.tgt_words):
        assert sent_words[sent_word_idx].lower() == 'the', f"In {cc.sent}, idx {sent_word_idx} is not 'the'"
        comp_word_idx = sent_word_idx + 1 # after the
        comparative_word_clean = trim_punctuation(sent_words[comp_word_idx])

        offset_start = cc.tgt_word_offsets[i][0]+4  # after the
        offset_end = offset_start + len(comparative_word_clean)
        # print(f"In {sent} looking for {comparative_word_clean} == {sent[offset_start:offset_end]}")
        # if not is_comparative(comparative_word_clean):
        #     print(f"In {cc.sent}, {comparative_word_clean} not comp")
        #     doc = nlp(comparative_word_clean)
        #     print([t.tag_ for t in doc])
        if not is_comparative_with_context(cc.sent, (offset_start, offset_end)):
            # print(offset_start, offset_end)
            print(f"In {cc.sent}, {comparative_word_clean} not comp")
            doc = nlp(comparative_word_clean)
            print([t.tag_ for t in doc])
