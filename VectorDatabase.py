from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# Vector Database
class VectorDatabase:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.collection = None
    
    def connect_milvus(self):
        connections.connect("default", host="localhost", port="19530")
    
    def create_collection(self):
        fields = [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.STRING, max_length=1024)
        ]
        schema = CollectionSchema(fields, description="Collection of embeddings with metadata")
        self.collection = Collection(name=self.collection_name, schema=schema)
        self.collection.create()
    
    def insert_vectors(self, embeddings, metadata):
        entities = [
            {"embedding": emb.tolist(), "metadata": meta} for emb, meta in zip(embeddings, metadata)
        ]
        self.collection.insert(entities)
        self.collection.load()
