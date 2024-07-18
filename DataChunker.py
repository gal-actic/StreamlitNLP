from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

class DataChunker:
    def __init__(self, data):
        self.data = data
        self.embeddings = None
    
    def chunk_data(self):
        # Use a pre-trained sentence embedding model
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        embeddings = model.encode(self.data, convert_to_tensor=True)
        
        # Perform hierarchical clustering based on cosine similarity
        similarity_matrix = cosine_similarity(embeddings)
        clustering = AgglomerativeClustering(n_clusters=None, affinity='cosine', linkage='average', distance_threshold=0.5)
        clusters = clustering.fit_predict(similarity_matrix)
        
        # Group sentences into chunks based on clusters
        chunks = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in chunks:
                chunks[cluster_id] = []
            chunks[cluster_id].append(self.data[idx])
        
        return chunks

