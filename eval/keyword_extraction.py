import numpy as np

from keybert import KeyBERT


def extract_most_common_keyword(ans, model):
    # Check for nan
    if isinstance(ans, float) and np.isnan(ans):
        return np.nan

    # If it is a list, return list of most common keywords
    if isinstance(ans, list):
        return [extract_most_common_keyword(x, model) for x in ans]

    # If the length is one (single letter / number)
    if len(ans) == 1:
        return ans

    # Otherwise, extract keywords and return
    try:
        return model.extract_keywords(ans, keyphrase_ngram_range=(1, 1), stop_words=None)[0][0]
    except IndexError:
        return ans


def extract_k_most_common_keywords_ngrams(ans, model):
    # Check for nan
    if isinstance(ans, float) and np.isnan(ans):
        return np.nan

    # Check for single-letter or single-number answers
    if len(ans) == 1:
        return ans

    # Otherwise, extract keywords of ngram ranges 1-4
    keywords1 = model.extract_keywords(ans, keyphrase_ngram_range=(1, 1), stop_words=None)
    keywords2 = model.extract_keywords(ans, keyphrase_ngram_range=(1, 2), stop_words=None)
    keywords3 = model.extract_keywords(ans, keyphrase_ngram_range=(1, 3), stop_words=None)
    keywords4 = model.extract_keywords(ans, keyphrase_ngram_range=(1, 4), stop_words=None)

    # Concatenate top 5 per ngram range
    result = [x[0] for x in keywords1[:5]]
    result.extend([x[0] for x in keywords2[:5]])
    result.extend([x[0] for x in keywords3[:5]])
    result.extend([x[0] for x in keywords4[:5]])

    # Sanity check
    if len(result) == 0:
        return ans
    return result


def extract_k_most_common_keyword(ans, model, k=5):
    # Check for nan
    if isinstance(ans, float) and np.isnan(ans):
        return np.nan

    # If the length is one (single letter / number)
    if len(ans) == 1:
        return ans

    # Otherwise, extract keywords and return
    try:
        ex = model.extract_keywords(ans, keyphrase_ngram_range=(1, 1), stop_words=None)
        return [x[0] for x in ex[:k]]
    except IndexError:
        return ans


def load_model_for_keywords():
    return KeyBERT("all-MiniLM-L6-v2")
