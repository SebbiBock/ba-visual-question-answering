from typing import List
from nltk.corpus import wordnet
from sentence_transformers import util as sentenceutil


def vqa_v2_accuracy(answer: str, annotated_answers: List[str]) -> float:
    """
        Compute the VQAv2 Accuracy for the given answer and the annotated answers.

        :param answer: The given answer, either human or by machines
        :param annotated_answers: The annotated answers of VQAv2
        :return: VQAv2 Accuracy in [0, 1]
    """

    return min(annotated_answers.count(answer) / 3, 1)


def accuracy(answer: str, annotated_answer: str) -> float:
    """
        Compute the Accuracy for the given answer and the annotated answers.

        :param answer: The given answer, either human or by machines
        :param annotated_answer: The annotated answer of VQAv2
        :return: Accuracy in {0, 1}
    """

    return 1 if answer.lower() == annotated_answer.lower() else 0


def wu_palmer_similarity(answer: str, annotated_answer: str) -> float:
    """
        As the Wu-Palmer-Similarity is only defined for single words, one has to average the score
        over all pair-wise computed Wu-Palmer-Similarities. -> Not a good measure

        :param answer: The given answer, either human or by machines
        :param annotated_answer: The annotated answer of VQAv2
        :return: The Wu-Palmer-Similarity in range [0, 1].
    """

    try:
        return wordnet.synsets(answer)[0].wup_similarity(wordnet.synsets(annotated_answer)[0])
    except IndexError:
        # print(f"No Wu-Palmer for {answer} or {annotated_answer}")
        pass
    return 0


def cosine_similarity(answer: str, annotated_answer: str, model):
    """
        Compute the cosine similarity between the embedding vector of the two given answers. The embeddings
        are retrieved using the SENTENCE-BERT SOTA-Transformer.

        NOTE: The spacy similarity

        doc1 = nlp(answer)
        doc2 = nlp(annotated_answer)
        return doc1.similarity(doc2)

        yields very similar results to the current approach.

        :param answer: The given answer, either human or by machines
        :param annotated_answer: The annotated answer of VQAv2
        :param model: The used SOTA-Model
        :return: Cosine Similarity in [0, 1]
    """

    # Encode sentences to get their embeddings
    embedding1 = model.encode(answer, convert_to_tensor=True)
    embedding2 = model.encode(annotated_answer, convert_to_tensor=True)

    # Compute and return similarity scores of two embeddings
    cosine_sim = sentenceutil.pytorch_cos_sim(embedding1, embedding2).item()
    cosine_sim = 1 if cosine_sim >= 1 else cosine_sim
    cosine_sim = -1 if cosine_sim <= -1 else cosine_sim
    return cosine_sim

