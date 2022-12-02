import json
import os
from typing import Dict, Iterable, List, Tuple, Union

import torch
from torch import Tensor, nn


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_mean_mark_tokens: bool = False,
    ):
        super(Pooling, self).__init__()

        self.config_keys = [
            'word_embedding_dimension', 'pooling_mode_cls_token',
            'pooling_mode_mean_tokens', 'pooling_mode_max_tokens',
            'pooling_mode_mean_mark_tokens',
            'pooling_mode_mean_sqrt_len_tokens'
        ]

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_mean_mark_tokens = pooling_mode_mean_mark_tokens

        pooling_mode_multiplier = sum([
            pooling_mode_cls_token, pooling_mode_max_tokens,
            pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens,
            pooling_mode_mean_mark_tokens
        ])
        self.pooling_output_dimension = (pooling_mode_multiplier *
                                         word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features['token_embeddings']
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']
        mark_token_ids = features['mark_token_ids']

        seq_length = token_embeddings.shape[1]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            token_embeddings[
                input_mask_expanded ==
                0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            token_ids = features['input_ids']
            meaningful_token_ids = [
                i.index(3) for i in token_ids.cpu().numpy().tolist()
            ]
            for i in meaningful_token_ids:
                attention_mask[:, i:] = 0
                attention_mask[:, :1] = 0

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(
                token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded,
                                       1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(
                    sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_mean_mark_tokens:

            marked_tokens = mark_token_ids.unsqueeze(-1).expand(
                token_embeddings.size()).float()

            sum_mask = marked_tokens.sum(1)

            filtered_features = torch.sum(token_embeddings * marked_tokens, 1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            output_vectors.append(filtered_features / sum_mask)

        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)
