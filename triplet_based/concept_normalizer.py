"""
Created on March 18 2021
@author: Dongfang Xu

Part of this library is based on sentence-transformers[https://github.com/UKPLab/sentence-transformers]
"""
import math
import queue

import numpy as np
import torch
import torch.multiprocessing as mp
import transformers
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity

import read_files as read


class ConceptNormalizer():
    """
    Loads or create a concept normalizer model, that can be used to map concepts/mentions to embeddings.
    :param model_name_or_path: Filepath of pre-trained LM or fine-tuned sentence-transformers. If it is a path for fine-tuned sentence-transformer, please also set sentence_transformer True.
    :sentence_transformer: This parameter can be used to create custom SentenceTransformer models from scratch.
    :search_over_synonyms: Whether to generate concept embeddings by averaging synonyms of that concept and then search over concept.
    """
    def __init__(
        self,
        model_name_or_path: str = None,
        sentence_transformer: bool = False,
        search_over_synonyms: bool = True,
    ):

        if sentence_transformer == False:
            ######## Load pre-trained models ########
            ######## word_embedding_model = models.BERT(path_to_BERT-based-models) #####
            ######## word_embedding_model = models.RoBERTa(path_to_RoBERTa-based-models) #####
            word_embedding_model = models.BERT(model_name_or_path)

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False)

            self.concept_normalizer = SentenceTransformer(
                modules=[word_embedding_model, pooling_model])

        else:
            #### load fine-tuned sentence-BERT models ####
            self.concept_normalizer = SentenceTransformer(model_name_or_path)

        self.search_over_synonyms = search_over_synonyms
        self.concept_mentions = {}
        self.concepts = []

    def generate_embeddings(self, ontology):

        for idx, [synonym, concept] in enumerate(ontology):
            read.add_dict(self.concept_mentions, concept, synonym)

        if len(self.concepts) == 0:
            self.concepts = self.concept_mentions.keys()

        self.synonyms = []
        self.concept_mention_idx = {}
        self.idx_to_concept = {}

        idx = 0
        for concept in self.concepts:
            concept_synonyms = list(set(self.concept_mentions[concept]))
            self.synonyms += concept_synonyms
            end = idx + len(concept_synonyms)
            for index in range(idx, end):
                self.idx_to_concept[int(index)] = concept
                self.concept_mention_idx[concept] = (idx, end)
            idx = end

        self.ontology_embedding = self.concept_normalizer.encode(self.synonyms)

        if self.search_over_synonyms == False:
            ontology_embedding_avg = []
            for concept in self.concepts:
                s, e = self.concept_mention_idx[concept]
                embedding_synonyms = self.ontology_embedding[s:e]

                ontology_embedding_avg.append(
                    np.mean(embedding_synonyms, axis=0))
            self.ontology_embedding_avg = np.asarray(ontology_embedding_avg)
            self.ontology_embedding = None

    def load_ontology(self, concept_file_path=None):
        if concept_file_path is not None:
            ontology = read.read_from_tsv(concept_file_path)
            self.generate_embeddings(ontology)
        else:
            raise ValueError("Please specify the path of ontology files")

    def add_terms(self, term_concept_pairs=[]):
        """
        term_concept_pairs is a list of 2-element tuples,
        [(syn_1, concept_1), (syn_2,concept_2),...]
        """
        ontology = [[item[0], item[1]] for item in term_concept_pairs]

        self.generate_embeddings(ontology)

    def normalize(self, mention, top_k):

        mention_embedding = self.concept_normalizer.encode(
            [mention], show_progress_bar=True)

        if self.search_over_synonyms:
            similarity_matrix = cosine_similarity(mention_embedding,
                                                  self.ontology_embedding)
        else:
            similarity_matrix = cosine_similarity(mention_embedding,
                                                  self.ontology_embedding_avg)
            # similarity_matrix = similarity_matrix.astype(np.float16)
        idx = np.argsort(similarity_matrix).astype(
            np.int32)[:, ::-1][:, :top_k]

        scores_pre = [row[idx[i]]
                      for i, row in enumerate(similarity_matrix)][0]
        concepts_pre = [[self.idx_to_concept[item] for item in row]
                        for row in idx][0]

        if self.search_over_synonyms:
            synonyms_pre = [[self.synonyms[item] for item in row]
                            for row in idx][0]
            predictions = [(cui, syn, score) for cui, syn, score in zip(
                concepts_pre, synonyms_pre, scores_pre)]
            return predictions
        else:
            predictions = [(cui, self.concept_mentions[cui][:2], score)
                           for cui, score in zip(concepts_pre, scores_pre)]
            return predictions
