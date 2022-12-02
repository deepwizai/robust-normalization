import read_files as read


def process_ontology():
    ontology = read.read_from_tsv("data/ontology.tsv")
    concept_mentions = {}

    for idx, [synonym, concept] in enumerate(ontology):
        read.add_dict(concept_mentions, concept, synonym)

    concepts = list(concept_mentions.keys())
    synonyms = []
    concept_mention_idx = {}
    idx = 0
    for concept in concepts:
        concept_synonyms = list(set(concept_mentions[concept]))
        synonyms += concept_synonyms
        end = idx + len(concept_synonyms)
        for index in range(idx, end):

            concept_mention_idx[concept] = (idx, end)
        idx = end

    synonyms = [[item] for item in synonyms]

    read.save_in_tsv("data/ontology/ontology_synonyms.tsv", synonyms)
    read.save_in_json("data/ontology/ontology_concept", concepts)
    read.save_in_json("data/ontology/ontology_concept_synonyms_idx",
                      concept_mention_idx)


process_ontology()


def dev_evaluator():

    ontology = read.read_from_tsv("data/ontology/ontology_synonyms.tsv")

    cui_mention_idx = read.read_from_json(
        "data/ontology/ontology_concept_synonyms_idx")

    corpus = {"doc_" + str(id): item[0] for id, item in enumerate(ontology)}

    read.save_in_json("data/evaluator_path/corpus", corpus)

    doc_id2mesh_all = {}
    mesh2doc_id_all = {}
    for key, item in cui_mention_idx.items():
        doc_id2mesh = {"doc_" + str(id): key for id in range(item[0], item[1])}
        doc_id2mesh_all.update(doc_id2mesh)
        mesh2doc_id = {
            key: ["doc_" + str(id) for id in range(item[0], item[1])]
        }
        mesh2doc_id_all.update(mesh2doc_id)

    dev_input = read.read_from_tsv("data/input_raw/dev.tsv")
    mentions = [item[0] for item in dev_input]

    query = {
        "q_" + str(id): item[0]
        for id, item in enumerate(dev_input) if item[1] != "CUI-less"
    }

    relevant_docs = {
        "q_" + str(id): mesh2doc_id_all[item[1]]
        for id, item in enumerate(dev_input) if item[1] != "CUI-less"
    }
    read.save_in_json("data/evaluator_path/dev_queries", query)
    read.save_in_json("data/evaluator_path/dev_relevant_docs", relevant_docs)

    for qid, item in query.items():

        text = [
            ontology[int(doc_id.split("_")[1])][0]
            for doc_id in relevant_docs[qid]
        ]
        print(item, text)


dev_evaluator()
