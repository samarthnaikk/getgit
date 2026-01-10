"""
Vector storage and retrieval system for RAG-based repository analysis.

Provides interfaces for storing embeddings and retrieving relevant chunks
based on semantic similarity to natural language queries.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import pickle
import os
from dataclasses import dataclass

from .chunker import Chunk


@dataclass
class RetrievalResult:
    """
    Result from a retrieval query.
    
    Attributes:
        chunk: The retrieved chunk
        score: Similarity score (higher is more similar)
        rank: Rank in the results (1-indexed)
    """
    chunk: Chunk
    score: float
    rank: int
    
    def __repr__(self):
        return f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, chunk={self.chunk})"


class VectorStore(ABC):
    """
    Abstract base class for vector storage systems.
    
    This abstraction allows for easy integration with different vector databases
    (e.g., FAISS, Pinecone, Weaviate, local numpy arrays).
    """
    
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """
        Add chunks and their embeddings to the store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (len(chunks), embedding_dim)
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query vector of shape (embedding_dim,)
            top_k: Number of results to return
        
        Returns:
            List of (chunk, score) tuples, sorted by score descending
        """
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load the vector store from disk."""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored vectors and chunks."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using numpy for similarity computation.
    
    Uses cosine similarity for retrieval. Suitable for small to medium-sized
    repositories. For large-scale use, consider FAISS or other optimized stores.
    """
    
    def __init__(self):
        """Initialize empty vector store."""
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def add_chunks(self, chunks: List[Chunk], embeddings: np.ndarray):
        """Add chunks and embeddings to the store."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Number of embeddings ({embeddings.shape[0]}) must match "
                f"number of chunks ({len(chunks)})"
            )
        
        if self.embeddings is None:
            self.embeddings = embeddings
            self.chunks = chunks
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.chunks.extend(chunks)
        
        # Normalize embeddings for cosine similarity
        self.embeddings = self._normalize(self.embeddings)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
        
        Returns:
            List of (chunk, score) tuples
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Normalize query
        query_norm = self._normalize(query_embedding.reshape(1, -1))[0]
        
        # Compute cosine similarity
        similarities = np.dot(self.embeddings, query_norm)
        
        # Get top-k indices
        top_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = [
            (self.chunks[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def save(self, filepath: str):
        """Save to disk using pickle."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
    
    def load(self, filepath: str):
        """Load from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.embeddings = data['embeddings']
    
    def clear(self):
        """Clear all data."""
        self.chunks = []
        self.embeddings = None
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        Args:
            vectors: Array of shape (n, d)
        
        Returns:
            Normalized array of same shape
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def __len__(self):
        """Return number of stored chunks."""
        return len(self.chunks)


class Retriever:
    """
    High-level retrieval interface combining embeddings and vector storage.
    
    This class provides the main API for RAG-based retrieval in GetGit.
    """
    
    def __init__(self, embedding_model, vector_store: Optional[VectorStore] = None):
        """
        Initialize retriever.
        
        Args:
            embedding_model: Instance of EmbeddingModel
            vector_store: Instance of VectorStore (defaults to InMemoryVectorStore)
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store or InMemoryVectorStore()
    
    def index_chunks(self, chunks: List[Chunk], batch_size: int = 32):
        """
        Index chunks for retrieval.
        
        Args:
            chunks: List of Chunk objects to index
            batch_size: Batch size for embedding generation
        """
        if not chunks:
            return
        
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.embed(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
    
    def retrieve(self, query: str, top_k: int = 5, 
                 filter_type: Optional[str] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a natural language query.
        
        Args:
            query: Natural language query string
            top_k: Number of results to return
            filter_type: Optional filter by chunk type (e.g., 'code_function')
        
        Returns:
            List of RetrievalResult objects, ranked by relevance
        """
        # Embed the query
        query_embedding = self.embedding_model.embed_single(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Apply filters if specified
        if filter_type:
            results = [
                (chunk, score) for chunk, score in results
                if chunk.chunk_type.value == filter_type
            ]
        
        # Limit to top_k
        results = results[:top_k]
        
        # Convert to RetrievalResult objects
        retrieval_results = [
            RetrievalResult(chunk=chunk, score=score, rank=i + 1)
            for i, (chunk, score) in enumerate(results)
        ]
        
        return retrieval_results
    
    def save(self, filepath: str):
        """
        Save the retriever state to disk.
        
        Args:
            filepath: Path to save the retriever
        """
        self.vector_store.save(filepath)
    
    def load(self, filepath: str):
        """
        Load the retriever state from disk.
        
        Args:
            filepath: Path to load the retriever from
        """
        self.vector_store.load(filepath)
    
    def clear(self):
        """Clear all indexed data."""
        self.vector_store.clear()
    
    def __len__(self):
        """Return number of indexed chunks."""
        return len(self.vector_store)
