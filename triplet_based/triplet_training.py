"""
This script trains sentence transformers with a batch hard loss function.

The TREC dataset will be automatically downloaded and put in the datasets/ directory

Usual triplet loss takes 3 inputs: anchor, positive, negative and optimizes the network such that
the positive sentence is closer to the anchor than the negative sentence. However, a challenge here is
to select good triplets. If the negative sentence is selected randomly, the training objective is often
too easy and the network fails to learn good representations.

Batch hard triplet loss (httpread_files

In a batch, it checks for sent1 with label A what is the other sentence with label A that is the furthest (hard positive)
which sentence with another label is the closest (hard negative example). It then tries to optimize this, i.e.
all sentences with the same label should be close and sentences for different labels should be clearly seperated.
"""

import argparse
import csv
import logging
import os
import random
import urllib.request
from collections import defaultdict
from datetime import datetime

# from BatchHardSoftMarginTripletLoss import BatchHardSoftMarginTripletLoss
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation, losses, models
from sentence_transformers.datasets import SentenceLabelDataset
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

import read_files as read

# from Pooling_custom import Pooling
# from SentenceLabelDateset_custom import SentenceLabelDataset
# from transformer_custom import Transformer


# Inspired from torchnlp
def read_dataset(train_data_path):

    data = csv.reader(open(os.path.join(train_data_path), encoding="utf-8"),
                      delimiter="\t",
                      quoting=csv.QUOTE_NONE)

    label_map = {}
    train_set = []
    guid = 0
    for line in data:
        # there is one non-ASCII byte: sisterBADBYTEcity; replaced with space
        text, label = line
        if label not in label_map:
            label_map[label] = len(label_map)

        label_id = label_map[label]
        guid += 1
        train_set.append(InputExample(guid=guid, texts=[text], label=label_id))

    return train_set


def model_training(
    train_data_path,
    evaluator_path,
    model_name,
    output_path,
    train_batch_size,
    num_epochs,
    samples_per_label,
):

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    output_path = (output_path + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    os.makedirs(output_path, exist_ok=True)

    # You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
    # model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/'

    ### Create a torch.DataLoader that passes training batch instances to our model

    logging.info("Loading training dataset")
    train_set = read_dataset(train_data_path)

    # Load pretrained model
    word_embedding_model = models.Transformer(model_name)
    # tokenizer_args={"additional_special_tokens": ['<e>', '</e>']})

    # word_embedding_model.auto_model.resize_token_embeddings(
    #     len(word_embedding_model.tokenizer))

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)
    # pooling_mode_mean_mark_tokens=True)

    # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=2048, activation_function=nn.Tanh())

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.max_seq_length = 16

    logging.info("Read concept normalization training dataset")

    #### try different sample size ####

    train_data_sampler = SentenceLabelDataset(
        examples=train_set, samples_per_label=samples_per_label)

    ##### Try whether shuffle  #####  By default, it shouldn't be shuffled every epoch

    train_dataloader = DataLoader(train_data_sampler,
                                  batch_size=train_batch_size,
                                  drop_last=True)

    ### Triplet losses ####################
    ### There are 4 triplet loss variants:
    ### - BatchHardTripletLoss
    ### - BatchHardSoftMarginTripletLoss
    ### - BatchSemiHardTripletLoss
    ### - BatchAllTripletLoss
    #######################################

    # train_loss = losses.BatchAllTripletLoss(model=model)
    #train_loss = losses.BatchHardTripletLoss(sentence_embedder=model)
    train_loss = losses.BatchHardSoftMarginTripletLoss(model)
    #train_loss = losses.BatchSemiHardTripletLoss(sentence_embedder=model)

    # evaluator = []

    logging.info("Read concept normalization val dataset")

    ir_queries = read.read_from_json(
        os.path.join(evaluator_path, "dev_queries"))
    ir_corpus = read.read_from_json(os.path.join(evaluator_path, "corpus"))
    ir_relevant_docs = read.read_from_json(
        os.path.join(evaluator_path, "dev_relevant_docs"))
    ir_evaluator_n2c2_dev = evaluation.InformationRetrievalEvaluator(
        ir_queries,
        ir_corpus,
        ir_relevant_docs,
        corpus_chunk_size=300000,
        name="evaluation_results",
        map_at_k=[1, 3, 5, 10],
        batch_size=1024,
        show_progress_bar=True)

    # evaluator.append(ir_evaluator_n2c2_dev)
    # Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
    # We optimize the model with respect to the score from the last evaluator (scores[-1])
    # seq_evaluator = evaluation.SequentialEvaluator(evaluator, main_score_function=lambda scores: scores[1])

    logging.info("Performance before fine-tuning:")
    ir_evaluator_n2c2_dev(model)

    # warmup_steps = int(
    #     len(train_dataset) * num_epochs / train_batch_size * 0.1
    # )  # 10% of train data
    warmup_steps = 0

    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        # evaluator = None,
        evaluator=ir_evaluator_n2c2_dev,
        output_path_ignore_not_empty=True,
        optimizer_params={
            'lr': 1e-4,
            'eps': 1e-6,
            'correct_bias': False
        },
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The training of the triplet network.')

    parser.add_argument('--model',
                        help='the direcotory of the BERT-based model',
                        required=True)

    parser.add_argument('--input_path',
                        help='the path of the input training data',
                        required=True)

    parser.add_argument('--evaluator_path',
                        help='the path of the evaluator, the dev dataset',
                        required=True)

    parser.add_argument('--output_path',
                        help='the direcotory to save the models',
                        required=True)

    parser.add_argument(
        '--train_batch_size',
        help='the training batch size, typically, larger is better',
        required=True)

    parser.add_argument('--epoch_size',
                        help='The number of epoch size',
                        required=True)

    parser.add_argument('--samples_per_label',
                        help='The number of instances for each concept. ',
                        required=True)

    args = parser.parse_args()
    model_name = args.model
    train_data_path = args.input_path
    evaluator_data_path = args.evaluator_path
    output_path = args.output_path
    train_batch_size = args.train_batch_size
    epoch_size = args.epoch_size
    samples_per_label = args.samples_per_label

    model_training(
        train_data_path,
        evaluator_data_path,
        model_name,
        output_path,
        train_batch_size,
        epoch_size,
        samples_per_label,
    )
