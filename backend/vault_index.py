import numpy as np
import faiss
import threading

class VaultIndex:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VaultIndex, cls).__new__(cls)
                cls._instance._init_index()
            return cls._instance

    def _init_index(self):
        self.dimension = 512
        # IndexFlatIP calculates Inner Product. For L2-normalized vectors, IP == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.faiss_id_to_user_id = {}
        self.current_id = 0
        self.index_lock = threading.RLock()

    def add_identity(self, user_id: str, embedding: np.ndarray):
        """
        Adds a single identity to the FAISS index.
        The embedding MUST be a 1D numpy array of shape (512,).
        """
        with self.index_lock:
            # Ensure shape and type
            if embedding.shape != (self.dimension,):
                import logging
                logging.getLogger("uvicorn.error").warning(f"VaultIndex: Skipped identity {user_id} due to dimension mismatch. Expected ({self.dimension},), got {embedding.shape}")
                return # Skip dimension mismatch silently to handle legacy embeddings
            
            vec = embedding.astype(np.float32).copy()
            # L2 normalize the vector for Inner Product to equal Cosine Similarity
            faiss.normalize_L2(vec.reshape(1, -1))
            
            self.index.add(vec.reshape(1, -1))
            self.faiss_id_to_user_id[self.current_id] = user_id
            self.current_id += 1

    def clear_index(self):
        """
        Clears the FAISS index. Thread-safe.
        """
        with self.index_lock:
            self.index.reset()
            self.faiss_id_to_user_id.clear()
            self.current_id = 0

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Searches the FAISS index for the top_k nearest neighbors.
        Returns a list of tuples: (user_id, cosine_similarity_score).
        """
        with self.index_lock:
            if self.index.ntotal == 0:
                return []
                
            if query_embedding.shape != (self.dimension,):
                raise ValueError(f"Query must have shape ({self.dimension},)")
                
            q_vec = query_embedding.astype(np.float32).copy()
            faiss.normalize_L2(q_vec.reshape(1, -1))
            
            # Bound top_k
            bounded_top_k = max(1, min(top_k, 100))
            
            # Search
            distances, indices = self.index.search(q_vec.reshape(1, -1), min(bounded_top_k, self.index.ntotal))
            
            results = []
            for i in range(len(indices[0])):
                faiss_id = indices[0][i]
                if faiss_id != -1:
                    user_id = self.faiss_id_to_user_id.get(faiss_id)
                    score = float(distances[0][i])
                    if user_id:
                        results.append((user_id, score))
                        
            return results

vault_index = VaultIndex()
