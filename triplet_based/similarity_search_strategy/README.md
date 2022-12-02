# Triplet-Search-ConNorm
Similarity Search Strategy

## Setup
* sample as before


## Getting Started

### input files
* Please refer process_input.py for how to generate input files for similarity search.
* process_sentence_corpus.py: generate embeddings for mention texts and concept texts.
* average_embeddings.py: generate concept embeddings by averaging text embeddings.
* similarity_search_single.py: similarity search over synonyms.
* similarity_search.py: similarity search over concepts.


### Code to run

* Generate embeddings for mentions from corpus:
```
python3.7 process_sentence_corpus.py \
--model models/PubMedBERT/ \
--model_type bert \
--sentences data/mention/mention.tsv \
--output output_path/mention_embeddings_pred
```

* Generate embeddings for every synonyms in an ontology:
```
python3.7 process_sentence_corpus.py \
--model models/PubMedBERT/ \
--model_type bert \
--sentences data/ontology/ontology_synonyms.tsv \
--output output_path/ontology_syn_embeddings
```

* Generate the concept embeddings by averaging the embeddings of all synonyms in an ontology:
```
python3.7  average_embeddings.py \
--synonyms_emebedding_path output_path/ontology_syn_embeddings.npy \
--concept_path data/ontology/ontology_concept \
--concept_synonym_idx_path data/ontology/ontology_concept_synonyms_idx \
--file_name output_path/ontology_con_embeddings
```

* Search concept using synonyms' embeddings:
```
python3.7 similarity_search_single.py \
--mention_embeddings_path output_path/mention_embeddings_pred.npy \
--synonym_ebmedding_path output_path/ontology_syn_embeddings.npy \
--concept_synonym_idx_path data/ontology/ontology_concept_synonyms_idx \
--top_k 32 \
--concept_pre_path output_path/concept_pre_syn \
--concept_score_pre_path output_path/concept_score_pre_syn
```

* Search concept using concept embeddings:
```
python3.7 similarity_search.py \
--mention_embeddings_path output_path/mention_embeddings_pred.npy \
--synonym_ebmedding_path output_path/ontology_con_embeddings.npy \
--concepts_path data/ontology/ontology_concept \
--top_k 32 \
--concept_pre_path output_path/concept_pre_con \
--concept_score_pre_path output_path/concept_score_pre_con
```
