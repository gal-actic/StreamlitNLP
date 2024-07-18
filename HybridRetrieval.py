from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import CountVectorizer
from rank_bm25 import BM25Okapi


# Hybrid Retrieval
class HybridRetrieval:
    def __init__(self, collection):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.bm25 = None  # Placeholder for BM25 model
        self.collection = collection
    
    def encode_query(self, query):
        inputs = self.tokenizer(query, return_tensors='pt')
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    def bm25_search(self, query):
        tokenized_corpus = [self.tokenizer(text)['input_ids'] for text in self.collection]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = self.tokenizer(query)['input_ids']
        bm25_scores = bm25.get_scores(tokenized_query)
        return bm25_scores
    
    def bert_search(self, query):
        query_vector = self.encode_query(query)
        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        results = self.collection.search(query_vector, "embedding", search_params, limit=len(self.collection))
        return results
    
    def combine_scores(self, bm25_scores, bert_scores):
        # Placeholder for combining scores
        combined_scores = [(bm25_scores[i], bert_scores[i]) for i in range(len(bm25_scores))]
        return combined_scores
    
    def rank_results(self, combined_scores, top_k):
        # Placeholder for ranking results based on combined scores
        sorted_results = sorted(combined_scores, key=lambda x: x[0] + x[1], reverse=True)[:top_k]
        return sorted_results
