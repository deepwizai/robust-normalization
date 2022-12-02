# Triplet-Search-ConNorm
The Triplet-Search-ConNorm library provides models for normalizing textual mentions of concepts to concepts in an ontology.

## Triplet-Trained Vector Space Model
The approach we use for concept normalization is a transformer based vector-space model, which encodes mentions and concepts via transformer networks that are trained via a triplet objective with online hard triplet mining.

> Dongfang Xu, and Steven Bethard. 2021.
> [Triplet-Trained Vector Space and Sieve-Based Search Improve Biomedical Concept Normalization](https://www.aclweb.org/anthology/2021.bionlp-1.2).
> In: Proceedings of the 20th Workshop on Biomedical Language Processing

To use the Triplet-Search-ConNorm, create an instance of `ConceptNormalizer`, and load an ontology from a tsv file (plese see the [ontology](https://github.com/dongfang91/Triplet-Search-ConNorm/blob/main/data/ontology/ontology.tsv))

```python3.8

>>> from ConceptNormalizer import ConceptNormalizer
>>> normalizer = ConceptNormalizer(model_name_or_path='path-to-bert')
>>> normalizer.load_ontology('path-to-ontology')
>>> normalizer.normalize("head spinning", top_k=4)
[('C0273239', 'head wound', 0.96794385), ('C0018670', 'HEAD', 0.96687186), ('C0018670', 'head', 0.96687186), ('C0230420', 'legs', 0.96390116)]
>>> normalizer.add_terms(term_concept_pairs=[("head spinning","C0012833")])
>>> normalizer.normalize("head spinning", top_k=4)
[('C0012833', 'head spinning', 0.99999934), ('C0273239', 'head wound', 0.96794385), ('C0018670', 'HEAD', 0.96687186), ('C0018670', 'head', 0.96687186)]
```



## Setup
* We recommend Python 3.7 or higher.
* The triplet search is implemented with PyTorch (1.7.1) using [transformers v4.4.1](https://github.com/huggingface/transformers), and [sentence-transformers v0.4.1.2](https://github.com/UKPLab/sentence-transformers)

## Getting Started

### Data format
 * Please see data/ontology/ontology.tsv, data/input_path/train.tsv for the input format: each row is a pair of mention and concept.
 * They are also used for training sentence-transformers.
 * process_input.py shows how to process ontology and generate input for batch processing;
 it also shows how to generate the evaluation files for the following triplet training.

### Pre-trained models
```
word_embedding_model = models.BERT(path_to_BERT-based-models)
pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),pooling_mode_mean_tokens=True)
concept_normalizer = SentenceTransformer(
                modules=[word_embedding_model, pooling_model])
```

### Fine-tuned sentence-transformer based models
```
concept_normalizer = SentenceTransformer(model_name_or_path)
```
 * SentenceTransformer model is fine-tuned using triplet_training.py

### Code to train sentence-transformer using triplet network
* code to run the triplet network for the concept normalization tasks using ncbi dataset:
```
python3.7 triplet_training.py \
--model bert-base-uncased \
--input_path data/input_path/train.tsv \
--evaluator_path data/evaluator_path/ \
--train_batch_size 1024 \
--samples_per_label 3 \
--epoch_size 3.0 \
--output_path /path/to/output/
```


### Similarity Search Strategy
plese see the [similarity_search_strategy](https://github.com/dongfang91/Triplet-Search-ConNorm/blob/main/similarity_search_strategy/))