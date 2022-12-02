import pandas as pd
from tqdm import tqdm 

from dataset_get import *
from candidate_gen import Gen_Candidate

# KB
train_dictionary = load_dictionary(dictionary_path="ncbi-disease/train_dictionary.txt") 
val_dictionary = load_dictionary(dictionary_path="ncbi-disease/dev_dictionary.txt") 
test_dictionary = load_dictionary(dictionary_path="ncbi-disease/test_dictionary.txt") 

dict_names, dict_id = [], []
for row in train_dictionary:
    for id in row[1].split('|'):
        dict_names.append(row[0])
        dict_id.append(id)
for row in val_dictionary:
    for id in row[1].split('|'):
        dict_names.append(row[0])
        dict_id.append(id)
for row in test_dictionary:
    for id in row[1].split('|'):
        dict_names.append(row[0])
        dict_id.append(id)

# mention
train_queries = load_queries(
        data_dir = "ncbi-disease/processed_train", 
        filter_composite=True,
        filter_duplicate=True,
        filter_cuiless=True
    )

gen_candidate = Gen_Candidate(dict_names)

KB_dict = {dict_id[i]: dict_names[i] for i in range(len(dict_id))}

print("Generating data as csv")

MENTIONS, CONCEPTS, LABELS = [], [], []
for row in tqdm(train_queries):
    query_name, query_id = row[0], row[1].split('|')[0]
    try:
        pos_concept = KB_dict[query_id]
    except:
        print("Concept ID corresponding to mention not found in KB")
        continue
        
    # Get Negative samples
    neg_concepts = gen_candidate.get_candidates(query_name)
    if pos_concept in neg_concepts:
        neg_concepts.remove(pos_concept)
    concepts = set([pos_concept]+neg_concepts)
    mentions, labels = [query_name]*len(concepts), [1]+[0]*(len(concepts)-1)
    CONCEPTS.extend(concepts)
    MENTIONS.extend(mentions)
    LABELS.extend(labels)

df = pd.DataFrame(
    {'mention': MENTIONS,
     'concept': CONCEPTS,
     'label': LABELS,
    })

df.to_csv("BERT-based-Ranking-for-Biomedical-Entity-Normalization/inp_data.csv", index=False)