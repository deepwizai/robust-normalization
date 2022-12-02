import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import read_files as read


def main(mention_embeddings_path, synonym_ebmedding_path,
         concept_synonym_idx_path, top_k, concept_pre_path,
         concept_score_pre_path):

    query = np.load(mention_embeddings_path)
    documents = np.load(synonym_ebmedding_path)

    concept_synonym_idx = read.read_from_json(concept_synonym_idx_path)
    id2concept = {
        int(i): cui
        for cui, item in concept_synonym_idx.items()
        for i in range(int(item[0]), int(item[1]))
    }

    similarity_matrix = cosine_similarity(query, documents)
    similarity_matrix = similarity_matrix.astype(np.float16)
    idx = np.argsort(similarity_matrix)
    idx = idx.astype(np.int32)
    top_k = int(top_k)
    idx = idx[:, ::-1][:, :top_k]

    concept_score_pre = [
        row[idx[i]] for i, row in enumerate(similarity_matrix)
    ]
    concept_pre = [[id2concept[item] for item in row] for row in idx]

    read.save_in_json(concept_pre_path, concept_pre)
    np.save(concept_score_pre_path, concept_score_pre)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--mention_embeddings_path',
                        help='the file path of the mention embeddings',
                        required=True)

    parser.add_argument('--synonym_ebmedding_path',
                        help='the file path of the synonym embeddings',
                        required=True)

    parser.add_argument('--concept_synonym_idx_path',
                        help='the file path of the concept synonym indexes',
                        required=True)

    parser.add_argument('--top_k',
                        help='save the top k synonyms in the output file',
                        required=True)

    parser.add_argument('--concept_pre_path',
                        help='the output file path of the predicted concepts',
                        required=True)

    parser.add_argument(
        '--concept_score_pre_path',
        help='the output file path of the scores of the predicted concepts',
        required=False,
        default="")

    args = parser.parse_args()
    mention_embeddings_path = args.mention_embeddings_path
    synonym_ebmedding_path = args.synonym_ebmedding_path
    concept_synonym_idx_path = args.concept_synonym_idx_path
    top_k = args.top_k
    concept_pre_path = args.concept_pre_path
    concept_score_pre_path = args.concept_score_pre_path

    main(mention_embeddings_path, synonym_ebmedding_path,
         concept_synonym_idx_path, top_k, concept_pre_path,
         concept_score_pre_path)
