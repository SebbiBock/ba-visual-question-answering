import spacy

import numpy as np

from spacy.lang.en.stop_words import STOP_WORDS
from spellchecker import SpellChecker
from typing import List, Union


KNOWNS = ["draftsman", "draftsmen", "idk", "stipley", "p", "foldable", "oxes", "mph", "Kiel", "hospitalroom", "", " ",
          "vantastico", "wakeboarding", "st", "str", "lawnchairs", "beachchairs", "banksy", "amazonas", "shipley", "t",
          "bc", "shapley", "dunkin", "kmh", "bokeh", "koln", "halfpipe", "praire"]


def postprocess(sentence: Union[List, str], nlp, spacy_stopwords=STOP_WORDS) -> Union[List, str]:
    """
        postprocess the given answer(s):

        (1) Replace & with and
        (2) Remove special characters
        (3) Tokenize
        (4) Lemmatize
        (5) Remove Stop Words
        (6) Replace numbers with their respective token
        (7) Return
    """

    # Check for nan
    if isinstance(sentence, float) and np.isnan(sentence):
        return np.nan

    # Recursively call the method if the given input is a list
    if isinstance(sentence, list):
        return [postprocess(x, nlp) for x in sentence]

    # Replace & with and
    sentence = sentence.replace("&", "and")

    # Remove special characters!
    sentence = sentence.replace("!", "")
    sentence = sentence.replace("'", "")
    sentence = sentence.replace("?", "")
    sentence = sentence.replace(".", "")
    sentence = sentence.replace(",", "")
    sentence = sentence.replace("-", "")
    sentence = sentence.replace("#", "")

    # Process and lowercase
    tokens = nlp(sentence)
    answer = " ".join([token.lemma_ for token in tokens if token not in spacy_stopwords])
    answer = lowercase(answer)

    # Replace numbers
    # TODO: Make this bettter so this only happens at word-level, so that e.g. 2023 is not replaced
    for numerical, number_word in enumerate([
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
        "thirteen"
    ]):
        answer = answer.replace(str(numerical), number_word)

    # Replace none with zero
    return answer.replace("none", "zero")


def lowercase(answer):
    if isinstance(answer, list):
        return [x.lower() for x in answer if not isinstance(x, float)]
    if isinstance(answer, str):
        return answer.lower()
    print("UNKNOWN ANSWER TYPE")
    return answer


def spell_check(ans, spell_check_model):
    if isinstance(ans, list):
        return [spell_check(x, spell_check_model) for x in ans]
    if isinstance(ans, str):
        spell_checked = []
        for word in ans.split(" "):
            corr = spell_check_model.correction(word)
            spell_checked.append(corr if corr is not None and corr != "None" else word)
            if corr != word and corr is not None and corr != "None":
                print("Spelling correction:", word, "->", corr)
        return " ".join(spell_checked)
    return ans


def load_models_for_postprocessing():

    # Load necessary models
    spell = SpellChecker()

    # Set known words for the spell checker
    spell.word_frequency.load_words(KNOWNS)
    spell.known(KNOWNS)

    # Load spacy model
    nlp = spacy.load('en_core_web_sm')

    return spell, nlp


ABSTRACT_CORRECT_ANSWERS = [
    "idk",
    "know",
    "dont",
    "don't",
    "do",
    "not",
    "cant",
    "cannot",
    "can",
    "know",
    "tell",
    "say",
    "possible",
    "impossible",
    "information",
    "unknown",
    "known",
    "no",
    "",
    " ",
]


def check_absurd_answer(ans):
    """
        Check if an absurd answer has been answered properly.
    """

    if isinstance(ans, float) and np.isnan(ans):
        return 1

    for word in ans.split(" "):
        if word in ABSTRACT_CORRECT_ANSWERS:
            return 1

    return 0