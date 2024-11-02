import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
    
    def add_documents(self, embeddings, documents):
        self.index.add(embeddings)
        self.documents.extend(documents)
    
    def search(self, query_embedding, k=10):
        scores, indices = self.index.search(query_embedding, k)
        return [(self.documents[i], scores[0][j]) for j, i in enumerate(indices[0])]
