import numpy as np
import pandas as pd

import data.loader as loader
import util.data_util as dutil

from eval.performance_metrics import accuracy, cosine_similarity, vqa_v2_accuracy, wu_palmer_similarity
from eval.postprocessing import check_absurd_answer, load_models_for_postprocessing, postprocess, spell_check
from eval.keyword_extraction import load_model_for_keywords, extract_k_most_common_keyword, extract_most_common_keyword

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def compute_performance():
    """
        Computes the performance scores [Accuracy, VQAv2-Accuracy, Wu-Palmer, Cosine Similarity] for all models and
        the human participants.
    """

    # Load models
    keyword_model = load_model_for_keywords()
    spellcheck_model, spacy_model = load_models_for_postprocessing()
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load in participant answers
    gaze_df, event_df, logger_df = loader.load_participant_data()
    question_ids = logger_df["question_id"].value_counts().index.tolist()[:10]

    # Get annotated answers
    annotated_answers = loader.load_annotated_answers(question_ids, single_answer=False)
    annotated_answers_single_answer = loader.load_annotated_answers(question_ids, single_answer=True)

    # Get model answers
    _, _, vilt_answer_df = loader.load_model_results("vilt")
    _, _, blip_answer_df = loader.load_model_results("blip")
    _, _, beit_answer_df = loader.load_model_results("beit3")

    # Get reasoning types
    a = loader.load_experiment_questions_for_group(1, only_values=False)
    b = loader.load_experiment_questions_for_group(2, only_values=False)
    reasoning_type_df = pd.concat([a, b])

    # Return lists
    accuracy_list = [[], [], [], []]
    vqav2_list = [[], [], [], []]
    cosine_list = [[], [], [], []]
    wu_palmer_list = [[], [], [], []]

    # Fill lists
    for q_id in tqdm(question_ids, ascii=True, desc="QUESTION-IDS:"):

        # Load in all answers and postprocess, as well as spell check them
        participant_answers = spell_check(postprocess(dutil.get_answers_for_question(logger_df, q_id), spacy_model), spellcheck_model)
        annotated_answer_list = spell_check(postprocess(annotated_answers[str(q_id)], spacy_model), spellcheck_model)
        single_annotated_answer = spell_check(postprocess(annotated_answers_single_answer[str(q_id)], spacy_model), spellcheck_model)
        vilt_answer = spell_check(postprocess(vilt_answer_df[vilt_answer_df["question_id"] == str(q_id)]["answer"].values[0], spacy_model), spellcheck_model)
        blip_answer = spell_check(postprocess(blip_answer_df[blip_answer_df["question_id"] == str(q_id)]["answer"].values[0], spacy_model), spellcheck_model)
        beit_answer = spell_check(postprocess(beit_answer_df[beit_answer_df["question_id"] == str(q_id)]["answer"].values[0], spacy_model), spellcheck_model)

        # Fill list with participant scores
        participant_scores_this_qid = [[], [], [], []]
        reasoning_type_qid = reasoning_type_df.isin([q_id]).any()[reasoning_type_df.isin([q_id]).any() == True].index[0]

        # Check if the answer is absurd
        if reasoning_type_qid == "absurd":
            participant_scores_this_qid[0] = participant_scores_this_qid[1] = participant_scores_this_qid[2] = participant_scores_this_qid[3] = [
                check_absurd_answer(x) for x in participant_answers
            ]

        # Otherwise, fill normally
        else:

            # Extract most common keyword for each vqav2 annotation
            annotations_common_keyword = extract_most_common_keyword(annotated_answer_list, keyword_model)

            for this_participant_answer in participant_answers:

                # Extract most common keywords of participants
                participant_common_keywords = extract_k_most_common_keyword(this_participant_answer, keyword_model)

                # Check for nan
                if isinstance(this_participant_answer, float) and np.isnan(this_participant_answer):
                    continue

                if len(participant_common_keywords) == 0:
                    print("EMPTY")
                    print(this_participant_answer)
                    print(q_id)
                    continue

                # Compute scores for all participant keywords and the annotation keywords
                vqav2 = max([vqa_v2_accuracy(keyword_answer, annotations_common_keyword) for keyword_answer in participant_common_keywords])
                acc = max([accuracy(keyword_answer, single_annotated_answer) for keyword_answer in participant_common_keywords])

                # Cosine similarity and wu-palmer similarity:
                cosine_sim = np.mean([
                    max([
                        cosine_similarity(keyword_answer, annotated_keyword, embedding_model) for keyword_answer in participant_common_keywords
                    ]) for annotated_keyword in annotations_common_keyword]
                )

                # Compute wu-palmer
                wu_palmer = np.mean([
                    max([
                        wu_palmer_similarity(keyword_answer, annotated_keyword) for keyword_answer in participant_common_keywords
                    ]) for annotated_keyword in annotations_common_keyword]
                )

                # Append results
                participant_scores_this_qid[0].append(acc)
                participant_scores_this_qid[1].append(vqav2)
                participant_scores_this_qid[2].append(cosine_sim)
                participant_scores_this_qid[3].append(wu_palmer)

        # Append human results
        accuracy_list[0].append(participant_scores_this_qid[0])
        vqav2_list[0].append(participant_scores_this_qid[1])
        cosine_list[0].append(participant_scores_this_qid[2])
        wu_palmer_list[0].append(participant_scores_this_qid[3])

        # Check if the answer is absurd
        if reasoning_type_qid == "absurd":
            for model_idx in [1, 2, 3]:
                for metric_lst in [accuracy_list, vqav2_list, cosine_list, wu_palmer_list]:
                    metric_lst[model_idx].append(check_absurd_answer([vilt_answer, blip_answer, beit_answer][model_idx - 1]))
            continue

        # Append ViLT results
        vilt_keyword = extract_most_common_keyword(vilt_answer, keyword_model)

        accuracy_list[1].append(accuracy(vilt_keyword, single_annotated_answer))
        vqav2_list[1].append(vqa_v2_accuracy(vilt_keyword, annotations_common_keyword))
        cosine_list[1].append(
            np.mean(
                [cosine_similarity(vilt_keyword, x, embedding_model) for x in annotations_common_keyword]
            )
        )
        wu_palmer_list[1].append(
            np.mean(
                [wu_palmer_similarity(vilt_keyword, x) for x in annotations_common_keyword]
            )
        )

        # Append Blip results
        blip_keyword = extract_most_common_keyword(blip_answer, keyword_model)

        accuracy_list[2].append(accuracy(blip_keyword, single_annotated_answer))
        vqav2_list[2].append(vqa_v2_accuracy(blip_keyword, annotations_common_keyword))
        cosine_list[2].append(
            np.mean(
                [cosine_similarity(blip_keyword, x, embedding_model) for x in annotations_common_keyword]
            )
        )
        wu_palmer_list[2].append(
            np.mean(
                [wu_palmer_similarity(blip_keyword, x) for x in annotations_common_keyword]
            )
        )

        # Append Beit results
        beit_keyword = extract_most_common_keyword(beit_answer, keyword_model)

        accuracy_list[3].append(accuracy(beit_keyword, single_annotated_answer))
        vqav2_list[3].append(vqa_v2_accuracy(beit_keyword, annotations_common_keyword))
        cosine_list[3].append(
            np.mean(
                [cosine_similarity(beit_keyword, x, embedding_model) for x in annotations_common_keyword]
            )
        )
        wu_palmer_list[3].append(
            np.mean(
                [wu_palmer_similarity(beit_keyword, x) for x in annotations_common_keyword]
            )
        )

        continue

    # Compute mean results and print
    human_accuracy = np.mean([item for sublist in accuracy_list[0] for item in sublist])
    vilt_accuracy = np.mean(accuracy_list[1])
    blip_accuracy = np.mean(accuracy_list[2])
    beit_accuracy = np.mean(accuracy_list[3])

    human_vqav2 = np.mean([item for sublist in vqav2_list[0] for item in sublist])
    vilt_vqav2 = np.mean(vqav2_list[1])
    blip_vqav2 = np.mean(vqav2_list[2])
    beit_vqav2 = np.mean(vqav2_list[3])

    human_cosine = np.mean([item for sublist in cosine_list[0] for item in sublist])
    vilt_cosine = np.mean(cosine_list[1])
    blip_cosine = np.mean(cosine_list[2])
    beit_cosine = np.mean(cosine_list[3])

    human_wu_palmer = np.mean([item for sublist in wu_palmer_list[0] for item in sublist])
    vilt_wu_palmer = np.mean(wu_palmer_list[1])
    blip_wu_palmer = np.mean(wu_palmer_list[2])
    beit_wu_palmer = np.mean(wu_palmer_list[3])

    print(f"Human Accuracy: {human_accuracy}")
    print(f"ViLT Accuracy: {vilt_accuracy}")
    print(f"Blip Accuracy: {blip_accuracy}")
    print(f"BEiT Accuracy: {beit_accuracy}")

    print(f"Human VQAv2: {human_vqav2}")
    print(f"ViLT VQAv2: {vilt_vqav2}")
    print(f"Blip VQAv2: {blip_vqav2}")
    print(f"BEiT VQAv2: {beit_vqav2}")

    print(f"Human Cosine: {human_cosine}")
    print(f"ViLT Cosine: {vilt_cosine}")
    print(f"Blip Cosine: {blip_cosine}")
    print(f"BEiT Cosine: {beit_cosine}")

    print(f"Human Wu-Palmer: {human_wu_palmer}")
    print(f"ViLT Palmer: {vilt_wu_palmer}")
    print(f"Blip Palmer: {blip_wu_palmer}")
    print(f"BEiT Palmer: {beit_wu_palmer}")

    accuracy_df = pd.DataFrame(accuracy_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    accuracy_df["vilt"] = accuracy_list[1]
    accuracy_df["blip"] = accuracy_list[2]
    accuracy_df["beit"] = accuracy_list[3]
    accuracy_df["question_id"] = question_ids
    accuracy_df["reasoning_type"] = accuracy_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    vqav2_df = pd.DataFrame(vqav2_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    vqav2_df["vilt"] = vqav2_list[1]
    vqav2_df["blip"] = vqav2_list[2]
    vqav2_df["beit"] = vqav2_list[3]
    vqav2_df["question_id"] = question_ids
    vqav2_df["reasoning_type"] = vqav2_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    wu_palmer_df = pd.DataFrame(wu_palmer_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    wu_palmer_df["vilt"] = wu_palmer_list[1]
    wu_palmer_df["blip"] = wu_palmer_list[2]
    wu_palmer_df["beit"] = wu_palmer_list[3]
    wu_palmer_df["question_id"] = question_ids
    wu_palmer_df["reasoning_type"] = wu_palmer_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    cosine_df = pd.DataFrame(cosine_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    cosine_df["vilt"] = cosine_list[1]
    cosine_df["blip"] = cosine_list[2]
    cosine_df["beit"] = cosine_list[3]
    cosine_df["question_id"] = question_ids
    cosine_df["reasoning_type"] = cosine_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    return accuracy_df, vqav2_df, wu_palmer_df, cosine_df


def compute_performance_no_preprocessing():

    # Load in embedding model for cosine sim
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load in participant answers
    gaze_df, event_df, logger_df = loader.load_participant_data()
    question_ids = logger_df["question_id"].value_counts().index.tolist()

    # Get annotated answers
    annotated_answers = loader.load_annotated_answers(question_ids, single_answer=False)
    annotated_answers_single_answer = loader.load_annotated_answers(question_ids, single_answer=True)

    # Get model answers
    _, _, vilt_answer_df = loader.load_model_results("vilt")
    _, _, blip_answer_df = loader.load_model_results("blip")
    _, _, beit_answer_df = loader.load_model_results("beit3")

    # Get reasoning types
    a = loader.load_experiment_questions_for_group(1, only_values=False)
    b = loader.load_experiment_questions_for_group(2, only_values=False)
    reasoning_type_df = pd.concat([a, b])

    # Return lists
    accuracy_list = [[], [], [], []]
    vqav2_list = [[], [], [], []]
    cosine_list = [[], [], [], []]
    wu_palmer_list = [[], [], [], []]

    # Fill lists
    for q_id in tqdm(question_ids, ascii=True, desc="QUESTION-IDS:"):

        # Load in all answers and postprocess, as well as spell check them
        participant_answers = dutil.get_answers_for_question(logger_df, q_id)
        annotated_answer_list = annotated_answers[str(q_id)]
        single_annotated_answer = annotated_answers_single_answer[str(q_id)]
        vilt_answer = vilt_answer_df[vilt_answer_df["question_id"] == str(q_id)]["answer"].values[0]
        blip_answer = blip_answer_df[blip_answer_df["question_id"] == str(q_id)]["answer"].values[0]
        beit_answer = beit_answer_df[beit_answer_df["question_id"] == str(q_id)]["answer"].values[0]

        # Fill list with participant scores
        participant_scores_this_qid = [[], [], [], []]
        reasoning_type_qid = reasoning_type_df.isin([q_id]).any()[reasoning_type_df.isin([q_id]).any() == True].index[0]

        # Check if the answer is absurd
        if reasoning_type_qid == "absurd":
            participant_scores_this_qid[0] = participant_scores_this_qid[1] = participant_scores_this_qid[2] = \
            participant_scores_this_qid[3] = [
                check_absurd_answer(x) for x in participant_answers
            ]

        # Otherwise, fill normally
        else:

            for this_participant_answer in participant_answers:

                # Check for nan
                if isinstance(this_participant_answer, float) and np.isnan(this_participant_answer):
                    continue

                # Compute scores for this participants answer
                vqav2 = vqa_v2_accuracy(this_participant_answer, annotated_answer_list)
                acc = accuracy(this_participant_answer, single_annotated_answer)
                cosine_sim = np.mean(
                    [cosine_similarity(this_participant_answer, ans, embedding_model) for ans in annotated_answer_list]
                )
                wu_palmer = np.mean(
                    [wu_palmer_similarity(this_participant_answer, ans) for ans in annotated_answer_list]
                )

                # Append results
                participant_scores_this_qid[0].append(acc)
                participant_scores_this_qid[1].append(vqav2)
                participant_scores_this_qid[2].append(cosine_sim)
                participant_scores_this_qid[3].append(wu_palmer)

        # Append human results
        accuracy_list[0].append(participant_scores_this_qid[0])
        vqav2_list[0].append(participant_scores_this_qid[1])
        cosine_list[0].append(participant_scores_this_qid[2])
        wu_palmer_list[0].append(participant_scores_this_qid[3])

        # Check if the answer is absurd
        if reasoning_type_qid == "absurd":
            for model_idx in [1, 2, 3]:
                for metric_lst in [accuracy_list, vqav2_list, cosine_list, wu_palmer_list]:
                    metric_lst[model_idx].append(
                        check_absurd_answer([vilt_answer, blip_answer, beit_answer][model_idx - 1]))
            continue

        # Append ViLT results
        accuracy_list[1].append(accuracy(vilt_answer, single_annotated_answer))
        vqav2_list[1].append(vqa_v2_accuracy(vilt_answer, annotated_answer_list))
        cosine_list[1].append(
            np.mean(
                [cosine_similarity(vilt_answer, x, embedding_model) for x in annotated_answer_list]
            )
        )
        wu_palmer_list[1].append(
            np.mean(
                [wu_palmer_similarity(vilt_answer, x) for x in annotated_answer_list]
            )
        )

        # Append Blip results
        accuracy_list[2].append(accuracy(blip_answer, single_annotated_answer))
        vqav2_list[2].append(vqa_v2_accuracy(blip_answer, annotated_answer_list))
        cosine_list[2].append(
            np.mean(
                [cosine_similarity(blip_answer, x, embedding_model) for x in annotated_answer_list]
            )
        )
        wu_palmer_list[2].append(
            np.mean(
                [wu_palmer_similarity(blip_answer, x) for x in annotated_answer_list]
            )
        )

        # Append Beit results
        accuracy_list[3].append(accuracy(beit_answer, single_annotated_answer))
        vqav2_list[3].append(vqa_v2_accuracy(beit_answer, annotated_answer_list))
        cosine_list[3].append(
            np.mean(
                [cosine_similarity(beit_answer, x, embedding_model) for x in annotated_answer_list]
            )
        )
        wu_palmer_list[3].append(
            np.mean(
                [wu_palmer_similarity(beit_answer, x) for x in annotated_answer_list]
            )
        )

        continue

    # Compute mean results and print
    human_accuracy = np.mean([item for sublist in accuracy_list[0] for item in sublist])
    vilt_accuracy = np.mean(accuracy_list[1])
    blip_accuracy = np.mean(accuracy_list[2])
    beit_accuracy = np.mean(accuracy_list[3])

    human_vqav2 = np.mean([item for sublist in vqav2_list[0] for item in sublist])
    vilt_vqav2 = np.mean(vqav2_list[1])
    blip_vqav2 = np.mean(vqav2_list[2])
    beit_vqav2 = np.mean(vqav2_list[3])

    human_cosine = np.mean([item for sublist in cosine_list[0] for item in sublist])
    vilt_cosine = np.mean(cosine_list[1])
    blip_cosine = np.mean(cosine_list[2])
    beit_cosine = np.mean(cosine_list[3])

    human_wu_palmer = np.mean([item for sublist in wu_palmer_list[0] for item in sublist])
    vilt_wu_palmer = np.mean(wu_palmer_list[1])
    blip_wu_palmer = np.mean(wu_palmer_list[2])
    beit_wu_palmer = np.mean(wu_palmer_list[3])

    print(f"Human Accuracy: {human_accuracy}")
    print(f"ViLT Accuracy: {vilt_accuracy}")
    print(f"Blip Accuracy: {blip_accuracy}")
    print(f"BEiT Accuracy: {beit_accuracy}")

    print(f"Human VQAv2: {human_vqav2}")
    print(f"ViLT VQAv2: {vilt_vqav2}")
    print(f"Blip VQAv2: {blip_vqav2}")
    print(f"BEiT VQAv2: {beit_vqav2}")

    print(f"Human Cosine: {human_cosine}")
    print(f"ViLT Cosine: {vilt_cosine}")
    print(f"Blip Cosine: {blip_cosine}")
    print(f"BEiT Cosine: {beit_cosine}")

    print(f"Human Wu-Palmer: {human_wu_palmer}")
    print(f"ViLT Palmer: {vilt_wu_palmer}")
    print(f"Blip Palmer: {blip_wu_palmer}")
    print(f"BEiT Palmer: {beit_wu_palmer}")

    accuracy_df = pd.DataFrame(accuracy_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    accuracy_df["vilt"] = accuracy_list[1]
    accuracy_df["blip"] = accuracy_list[2]
    accuracy_df["beit"] = accuracy_list[3]
    accuracy_df["question_id"] = question_ids
    accuracy_df["reasoning_type"] = accuracy_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    vqav2_df = pd.DataFrame(vqav2_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    vqav2_df["vilt"] = vqav2_list[1]
    vqav2_df["blip"] = vqav2_list[2]
    vqav2_df["beit"] = vqav2_list[3]
    vqav2_df["question_id"] = question_ids
    vqav2_df["reasoning_type"] = vqav2_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    wu_palmer_df = pd.DataFrame(wu_palmer_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    wu_palmer_df["vilt"] = wu_palmer_list[1]
    wu_palmer_df["blip"] = wu_palmer_list[2]
    wu_palmer_df["beit"] = wu_palmer_list[3]
    wu_palmer_df["question_id"] = question_ids
    wu_palmer_df["reasoning_type"] = wu_palmer_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    cosine_df = pd.DataFrame(cosine_list[0], columns=[f"Participant_{x + 1}" for x in range(10)])
    cosine_df["vilt"] = cosine_list[1]
    cosine_df["blip"] = cosine_list[2]
    cosine_df["beit"] = cosine_list[3]
    cosine_df["question_id"] = question_ids
    cosine_df["reasoning_type"] = cosine_df["question_id"].apply(lambda x: reasoning_type_df.isin([x]).any()[reasoning_type_df.isin([x]).any() == True].index[0])

    return accuracy_df, vqav2_df, wu_palmer_df, cosine_df