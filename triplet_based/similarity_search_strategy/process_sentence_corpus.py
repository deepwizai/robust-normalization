import argparse

import numpy as np
from Pooling_custom import Pooling as Pooling
from sentence_transformers import SentenceTransformer, models

import read_files as read


def main(model_path, model_type, synonyms_path, output_path):

    #### Read sentence courpus.  output: list of sentences ####
    synonyms = read.read_from_tsv(synonyms_path)
    synonyms = [item for row in synonyms for item in row]
    print("The # of synonyms in an ontology: ", len(synonyms))
    print("The first 10 synonyms: ", synonyms[:10])

    if model_type.lower() in ["bert"]:
        # Load pretrained model
        word_embedding_model = models.Transformer(model_path)

        pooling_model = Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False)

        embedder = SentenceTransformer(
            modules=[word_embedding_model, pooling_model])

        #### load sentence BERT models and generate sentence embeddings ####
    else:
        #### load sentence BERT models and generate sentence embeddings ####
        embedder = SentenceTransformer(model_path)

    embedder.max_seq_length = 16
    sentences_embedding = embedder.encode(sentences,
                                          batch_size=1000,
                                          show_progress_bar=True,
                                          num_workers=4)

    read.create_folder(output_path)

    np.save(output_path, sentences_embedding)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        'Generate sentence embedding for each sentence in the sentence corpus '
    )

    parser.add_argument('--model',
                        help='the direcotory of the model',
                        required=True)

    parser.add_argument(
        '--model_type',
        help=
        'the type of the model, sentence_bert after triplet training or just pre-trained bert such as PubMedBert',
        required=True)

    parser.add_argument('--synonyms',
                        help='the file of all synonyms',
                        required=True)

    parser.add_argument('--output_path',
                        help='the direcotory of output synonyms embeddings',
                        required=True)

    args = parser.parse_args()
    model_path = args.model
    model_type = args.model_type
    synonyms_path = args.synonyms
    output_path = args.output_path

    main(model_path, model_type, synonyms_path, output_path)
