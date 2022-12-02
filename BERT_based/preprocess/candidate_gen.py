from gensim import corpora
from gensim.summarization import bm25

def simple_tok(sent:str):
    return sent.split()

class Gen_Candidate():
    """
    Generates n candidates for each entity input
    Params:
        corpus: List[str]
    """
    def __init__(self, docs):
        self.docs = docs
        self.texts = [simple_tok(doc) for doc in docs]
        self.dictionary = corpora.Dictionary(self.texts)
        self.bm25_obj = self.create_KB(docs)
        
    def create_KB(self, docs):
        corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        bm25_obj = bm25.BM25(corpus)
        return bm25_obj
    
    def get_candidates(self, query, n=10):
        query_doc = self.dictionary.doc2bow(query.split())
        scores = self.bm25_obj.get_scores(query_doc)
        best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        best_docs = list(set([self.docs[id] for id in best_idx]))
        return best_docs