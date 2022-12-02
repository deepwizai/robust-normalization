import argparse
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import read_files as read


def main(syn_path, cui_path, cui_idx_path, file_name):
    embeddings = np.load(syn_path)
    cuis = read.read_from_json(cui_path)
    cui_idx = read.read_from_json(cui_idx_path)
    avg = []
    for cui in cuis:
        s, e = cui_idx[cui]
        embedding_syn = embeddings[s:e]
        avg.append(np.mean(embedding_syn, axis=0))
    avg = np.asarray(avg)

    read.create_folder(file_name)
    np.save(file_name, avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--synonyms_emebedding_path',
                        help='the direcotory of the model',
                        required=True)

    parser.add_argument('--concept_path',
                        help='the path of the ontology concept file',
                        required=True)

    parser.add_argument('--concept_synonym_idx_path',
                        help='the path of the ontology concept idx path',
                        required=True)

    parser.add_argument('--output',
                        help='the path of the sentence corpus',
                        required=True)

    args = parser.parse_args()
    syn_embeddings_path = args.synonyms_emebedding_path
    cui_path = args.concept_path
    cui_idx_path = args.concept_synonym_idx_path
    output_path = args.output_path
    main(syn_embeddings_path, cui_path, cui_idx_path, output_path)
