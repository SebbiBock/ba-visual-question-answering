import os
import random
import re
import spacy
import sys

import pickle
import numpy as np
import pandas as pd

import data.loader as loader

from pathlib import Path
from typing import Dict, Tuple


########### DICTIONARY ###########

# List of categories
CATEGORIES = ["object presence",
              "subordinate object recognition",
              "counting",
              "color attributes",
              "other attributes",
              "activity recognition",
              "sport recognition",
              "positional reasoning",
              "scene classification",
              "sentiment understanding",
              "object utilities and affordances",
              "reading",
              "absurd"
             ]

# List of question words: Sport
SPORT_QUESTIONS = [
    "play",
    "throw",
    "catch",
    "game",
]

# List of answers: Sport
# Either: One of the ten most common sports or common sport related words
SPORT_ANSWERS = [
    "action",
    "ball",
    "athletics",
    "baseball",
    "football",
    "soccer",
    "cricket",
    "hockey",
    "tennis",
    "volleyball",
    "rugby",
    "golf",
    "badminton",
    "sport",
    "swim",
]

# List of answer words: Activities
ACTIVITY_QUESTIONS = [
    "do"
]

# List of answers: Activities
ACTIVITY_ANSWERS = [
    "chores",
    "cook",
    "sport",
    "play",
    "dance",
    "clean",
    "study",
    "read",
    "sport",
    "draw",
    "write",
    "swim",
    "work",
    "speaking",
    "ride",
    "nothing",
    "something",
    "run",
    "walk"
]

# List of questions and answers: Scene classification
SCENE_CLASSIFICATION_WORDS = [
    "sun",
    "rain",
    "warm",
    "cold",
    "freeze",
    "hot",
    "snow",
    "cloud",
]

# List of sentiment understaind questions and answers:
SENTIMENT_UNDERSTANDING_WORDS = [
    "feel",
    "happy",
    "sad",
    "unhappy",
    "glad",
    "annoyed",
    "angry",
    "neutral",
    "positive",
    "negative",
    "interested"
]


# List of other attributes
OTHER_ATTRIBUTES = [
    "big",
    "small",
    "long",
    "short",
    "tiny",
    "shiny",
    "flat",
    "huge",
    "slow",
    "fast",
]

################################


def get_absurd_questions(fixed_questions: bool = True, top_k_questions: int = 100):
    """
        This method is used to retrieve questions of reasoning type "absurd" from the VQAv2 validation set. This
        reasoning category needs its own function since it is not possible to capture it via keywords and regular
        expressions, but rather needs human judgement. Therefore, the VQAv2 annotations are loaded, and the answer
        confidences that the crowdworkers had to note are sorted to get the questions with the least agreed upon
        confidence. If fixed_questions is false, #top_k_questions of the top absurd questions are returned. If fixed
        questions is set to true, ten manually chosen absurd questions from these questions are returned.

        :param fixed_questions: Whether the manually annotated questions are to be returned [True] or not [False]
        :param top_k_questions: How many of the top k questions are to be returned, if fixed_questions is False
        :return: List of question ids of the most absurd questions
    """

    # If fixed questions are desired: Return the ten pre-chosen ones
    if fixed_questions:
        return [
            543065008,
            81054002,
            290704003,
            309434004,
            242870002,
            43433001,
            501443001,
            518224000,
            379022000,
            551692022
        ]

    # Load all answers
    all_answers = loader.load_annotated_answers(None)

    # Build matrix with question ids and answer confidences
    question_confidence_matrix = []

    # For every question
    for question in all_answers:

        # Get all its answer confidences
        confidences = []
        for answer in question["answers"]:
            confidences.append(answer["answer_confidence"])

        # Count occurrences of "maybe" and "no"
        unique, counts = np.unique(confidences, return_counts=True)

        yes_count = counts[list(unique).index("yes")] if "yes" in unique else 0
        no_count = counts[list(unique).index("no")] if "no" in unique else 0
        maybe_count = counts[list(unique).index("maybe")] if "maybe" in unique else 0

        question_confidence_matrix.append([question["question_id"], yes_count, maybe_count, no_count])

    # Turn into array
    matrix = np.array(question_confidence_matrix)

    # Sort the matrix by the no counts
    sorted_matrix = matrix[matrix[:, -1].argsort()[::-1]]

    return sorted_matrix[:top_k_questions, 0]


def reasoning_category(question: str, answer: str, tokenizer):
    """
        This method classifies the given question into any of the given thirteen reasoning categories via the use of
        keywords or regular expressions. If no criteria matches, "n/a" is returned. For this, the question and the answer
        is linguistically preprocessed using spaCy's tokenizer. Then, an order of categories is defined (from fine to
        broad), and the first category expression that the question satisfies is returned.

        ------------------------------------------------------
        POSSIBLE CATEGORIES (Kafle & Kanan, 2017) + READING
        ------------------------------------------------------

        1. Object Presence (e.g., ‘Is there a cat in the image?’)
        2. Subordinate Object Recognition (e.g., ‘What kind of furniture is in the picture?’)
        3. Counting (e.g., ’How many horses are there?’)
        4. Color Attributes (e.g., ‘What color is the man’s tie?’)
        5. Other Attributes (e.g., ‘What shape is the clock?’)
        6. Activity Recognition (e.g., ‘What is the girl doing?’)
        7. Sport Recognition (e.g.,‘What are they playing?’)
        8. Positional Reasoning (e.g., ‘What is to the left of the man on the sofa?’)
        9. Scene Classification (e.g., ‘What room is this?’)
        10. Sentiment Understanding (e.g.,‘How is she feeling?’)
        11. Object Utilities and Affordances (e.g.,‘What object can be used to break glass?’)
        12. Reading (e.g., What does the sign say?: Not from Kafle & Kanan, but rather VQA-MHUG)
        13. Absurd (i.e., Nonsensical queries about the image)

        :param question: The question to assign the reasoning category to
        :param answer: The answer to the given question
        :param tokenizer: The spaCy tokenizer to use
        :return: The object category as string, or "n/a"
    """

    ### PREPROCESSING ###
    question = question.replace("?", "")

    # Preprocessing: Use spacy to process the question and answer
    question_doc = tokenizer(question)
    question_tokens = [str(token) for token in question_doc]
    question_lemmas = [str(token.lemma_) for token in question_doc]
    question_lemmas_joined = " ".join(question_lemmas)
    question_pos = [str(token.pos_) for token in question_doc]

    answer_doc = tokenizer(answer)
    answer_tokens = [str(token) for token in answer_doc]
    answer_lemmas = [str(token.lemma_) for token in answer_doc]

    ### CATEGORIES ###

    ### 12: READING ###

    # What is written, printed, what does ... say/mean, what word|sentence|letter
    if re.search("^what be (write|print){1}", question_lemmas_joined) is not None:
        return CATEGORIES[11]

    if re.search("^what do .* (say|mean)", question_lemmas_joined) is not None:
        return CATEGORIES[11]

    if re.search("^what *.* (word|sentence|name|letter|number) .*", question_lemmas_joined) is not None:
        return CATEGORIES[11]

    ### 3: COUNTING ###
    if "how" in question_lemmas and "many" in question_lemmas:
        return CATEGORIES[2]

    ### 4: COLOR ATTRIBUTES ###
    if "color" in question_lemmas:
        return CATEGORIES[3]

    ### 2: SUBORDINATE OBJECT RECOGNITION ###

    # What kind of xyz is in the image?
    if re.search("^what .* (is|be){1} in the (image|picture|scene){1}", question_lemmas_joined) is not None:
        return CATEGORIES[1]

    # What kind of .* (is|be|are)
    if re.search("^what (kind|type){1} of .*", question_lemmas_joined) is not None:
        if "VERB" in question_pos[3:] or "AUX" in question_pos[3:]:
            return CATEGORIES[1]

    # What xyz is this?
    if re.search("^what .* (is|be){1} (here|this|depict|show){1}", question_lemmas_joined) is not None:
        return CATEGORIES[1]

    # What kind of * is .*
    if re.search("^what (kind|type){1} of .* (is|are|be){1}", question_lemmas_joined) is not None:
        return CATEGORIES[1]

    ### 7: SPORT RECOGNITION ###

    for ans in SPORT_ANSWERS:
        if ans in question_lemmas:
            return CATEGORIES[6]

    for lem in question_lemmas:
        if lem in SPORT_QUESTIONS:
            for ans_lem in answer_lemmas:
                if ans_lem in SPORT_ANSWERS:
                    return CATEGORIES[6]

    ### 9. SCENE CLASSIFICATION ###

    # Where is this, where can this be found?
    if re.search("^where (is|be|are|can){1} (this){1} .*", question_lemmas_joined) is not None:
        return CATEGORIES[8]

    # Is/Are this/they/he/she/it in NOUN?
    if re.search("^(is|are|be) (this|they|he|she|it){1} .* in .*", question_lemmas_joined) is not None:
        if question_pos[-1] == "NOUN":
            return CATEGORIES[8]

    # Is x at y?
    if re.search("^(is|be) .* at .*", question_lemmas_joined) is not None:
        return CATEGORIES[8]

    # Weather stuff
    if "weather" in question_lemmas:
        return CATEGORIES[8]

    for w in SCENE_CLASSIFICATION_WORDS:
        if w in question_lemmas or w in answer_lemmas:
            return CATEGORIES[8]

    ### 8. POSITIONAL REASONING ###

    # What is to the left/right ...?
    if re.search("^what (is|be|are){1} *(to|on)* *the* *(left|right){1} .*", question_lemmas_joined) is not None:
        return CATEGORIES[7]

    # What is above/below?
    if re.search(
            "^what (is|be|are){1} *.* *(on|above|below|under|behind|in front|on top|inside|outside|back|background|foreground){1} .*",
            question_lemmas_joined) is not None:
        return CATEGORIES[7]

    if re.search("^(is|be|are){1} the{1} *.* *(above|below|behind|in front|on top|inside|outside){1} .*",
                 question_lemmas_joined) is not None:
        return CATEGORIES[7]

    # What is in the upper/left/lower ... corner / edge ...?
    if re.search("^what (is|are|be){1} .* (upper|lower|left|right)+ (corner|edge|side){1} .*",
                 question_lemmas_joined) is not None:
        return CATEGORIES[7]

    # Where is he .*? Where is the cow? ...
    if re.search("^where (be|is|are){1} (he|she|they|it|the){1} .*", question_lemmas_joined) is not None:
        return CATEGORIES[7]

    # Where does x start / end?
    if re.search("^where do *.* *(start|end){1} *.*", question_lemmas_joined) is not None:
        return CATEGORIES[7]

    # In which direction is ... looking?
    if re.search("in which direction *.*", question_lemmas_joined) is not None:
        return CATEGORIES[7]

    ### 10. SENTIMENT UNDERSTANDING ###
    if "feel" in question_lemmas:
        return CATEGORIES[9]

    for w in SENTIMENT_UNDERSTANDING_WORDS:
        if w in question_lemmas or w in answer_lemmas:
            return CATEGORIES[9]

    ### 11. OBJECT UTILITIES AND AFFORDANCES ###
    if "why" in question_lemmas:
        return CATEGORIES[10]

    # (What object / Can ) .*  (can) be used to *?
    if re.search("^(what|can) .* be use .*", question_lemmas_joined) is not None:
        return CATEGORIES[10]

    # Can * be *
    if re.search("^can .* be .*", question_lemmas_joined) is not None:
        return CATEGORIES[10]

    if "useful" in question_lemmas:
        return CATEGORIES[10]

    ### 6: ACTIVITY RECOGNITION ###

    # What (is|be|are) do?
    if re.search("^what (is|be|are){1} .* do .*", question_lemmas_joined) is not None:
        return CATEGORIES[5]

    # What (do|is|are) ... VERB?
    if re.search("^what (do|is|be|are){1} .*", question_lemmas_joined) is not None:
        if question_pos[-1] == "VERB" or question_pos[-1] == "AUX":
            return CATEGORIES[5]

    # Is/Are .* verb?
    if re.search("^(is|are|be) (the|this|they|he|she|it|that|those){1} .*", question_lemmas_joined) is not None:
        if question_pos[-1] == "VERB" or question_pos[-1] == "AUX":
            return CATEGORIES[5]

    for lem in question_lemmas:
        if lem in ACTIVITY_QUESTIONS:
            for ans_lem in answer_lemmas:
                if ans_lem in ACTIVITY_ANSWERS:
                    return CATEGORIES[5]

    ### 5: OTHER ATTRIBUTES ###

    # What * is the *
    if re.search("^what .* (is|be|are){1} the .*", question_lemmas_joined) is not None:
        return CATEGORIES[4]

    # Is this ADJ .*?
    if re.search("^(is|be|are) (this|those|that|he|she|they|the|anything|everything|something){1} .*",
                 question_lemmas_joined) is not None:
        if "ADJ" in question_pos[2:]:
            return CATEGORIES[4]

    # If shape or form is in the question
    if "shape" in question_lemmas or "form" in question_lemmas:
        return CATEGORIES[4]

    if re.search("(make|create|consist){1} (of|from){1}", question_lemmas_joined) is not None:
        return CATEGORIES[4]

    if question_lemmas[0] == "be" or question_lemmas[0] == "how":
        for q in question_lemmas[1:]:
            if q in OTHER_ATTRIBUTES:
                return CATEGORIES[4]

    ### 13. ABSURD ###

    # Not determined here, but via another approach

    ### 1: OBJECT PRESENCE ###

    # Moved to the end since this has the broadest expressions

    # Begins with is / are, followed by there and any sequence of characters
    if re.search("^(is|are|be) there .*", question_lemmas_joined) is not None:
        return CATEGORIES[0]

    # Begins with is / are, followed by any tokens and then followed by either on/ in and the
    if re.search("^(is|are|be) .* on|in the *", question_lemmas_joined) is not None:
        return CATEGORIES[0]

    if re.search("^can .*(be|you){1} see *.*", question_lemmas_joined) is not None:
        return CATEGORIES[0]

    if re.search("^(is|are|be) this .*", question_lemmas_joined) is not None:
        return CATEGORIES[0]

    if re.search("^what be", question_lemmas_joined) is not None:
        return CATEGORIES[0]

    # What noun be|do
    if re.search("^what .* (be|do){1} .*", question_lemmas_joined) is not None:
        if question_pos[1] == "NOUN":
            return CATEGORIES[0]

    ### IF NOTHING IS FOUND ###
    return "n/a"


def annotate_questions_with_reasoning_type() -> Dict[str, str]:
    """
        This method loads in all questions and answers of the VQAv2 validation set and annotates them with the
        reasoning category proposed by Kafle & Kanan (2017), with the extension of the "Reading" category that
        was proposed by Sood et al (2021). The final result is then stored as a dictionary which is finally
        saved to the current working directory and returned.

        :return: The type dict indicating the annotated reasoning type to corresponding question IDs.
    """

    # Load in necessary data
    questions = loader.load_questions(None)
    answers = loader.load_annotated_answers([q["question_id"] for q in questions])
    q_ans_dict = {q["question_id"]: (q["question"], ans) for q, ans in zip(questions, answers.values())}

    # Load tokenizer
    tokenizer = spacy.load("en_core_web_sm")

    # Construct dictionary
    type_dict = {}

    # Iterate over all questions and ansers and retrieve category
    for q_id, (question, answer) in q_ans_dict.items():
        category = reasoning_category(question, answer, tokenizer)
        if category != "n/a":
            type_dict[q_id] = category

    # Get 100 absurd questions
    absurd_questions = get_absurd_questions(False, 100)
    for q_id in absurd_questions:
        if q_id not in type_dict.keys():
            type_dict[q_id] = "absurd"

    # Save the dictionary
    with open(Path(loader.PATH_DICT["REASONING_TYPES_PATH"]), "wb") as rtf:
        pickle.dump(type_dict, rtf)

    # Return the type dict
    return type_dict


def choose_questions_at_random(type_dict: Dict[str, str], num_questions: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Method to choose x random questions for each reasoning type that are then split into two groups. While choosing
        these questions at random, this method ensures that the number of yes and no answers over all yes/no questions
        is balanced 50/50. The resulting two pd.DataFrames contain the reasoning types as columns and each entry
        represents one question ID belonging to that reasoning type.

        :param type_dict: The dictionary containing the reasoning types for the question IDs
        :param num_questions: The number of questions to choose for each reasoning type, has to be dividable by 2.
        :return: A Tuple of two pd.DataFrames, one for each group.
    """

    if num_questions % 2 != 0:
        raise ValueError("The number of questions to choose has to be dividable by two.")

    # Incredibly inefficient, but: iterate until the requirement of no == yes is met:
    while True:

        # Final return list
        final_list = np.zeros((num_questions, len(CATEGORIES)))

        # Loop through all categories
        for cat_idx, category in enumerate(CATEGORIES):

            list_this_category = []

            for q_id, cat in type_dict.items():
                if category == cat:
                    list_this_category.append(q_id)

            # Shuffle the list
            random.shuffle(list_this_category)

            if len(list_this_category) > 0:
                # Take the first ten elements
                final_list[:, cat_idx] = list_this_category[:num_questions]

        # Insert the chosen absurd questions
        final_list[:, -1] = get_absurd_questions()

        # Load in answers
        annotations = loader.load_annotated_answers(final_list.flatten().astype(int), single_answer=True)

        # Get yes and no answers
        yes_count = 0
        no_count = 0
        for _, ans in annotations.items():
            if ans == "yes":
                yes_count += 1
            if ans == "no":
                no_count += 1

        # If they are equal: break
        if yes_count == no_count:
            break

    # Create two dataframes and return
    group1 = pd.DataFrame(final_list[:int(num_questions / 2)], columns=CATEGORIES, dtype=int)
    group2 = pd.DataFrame(final_list[int(num_questions / 2):], columns=CATEGORIES, dtype=int)

    return group1, group2


if __name__ == '__main__':

    # Create annotations
    annotated_dict = annotate_questions_with_reasoning_type()

    # Sample 10 questions for every reasoning category and save them, as well
    group1, group2 = choose_questions_at_random(annotated_dict, 10)

    # Save the group 1 pd.DataFrame
    with open("group_1_questions.pkl", "wb") as f:
        pickle.dump(group1, f)

    # Save the group 2 pd.DataFrame
    with open("group_2_questions.pkl", "wb") as f:
        pickle.dump(group2, f)
