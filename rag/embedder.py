"""
Embedding model abstraction for converting text chunks into vector representations.

Provides a pluggable interface for different embedding models, with a default
implementation using sentence-transformers.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.
    
    This abstraction allows for easy swapping of different embedding models
    without changing the retrieval system.
    """
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings into vector representations.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string.
        
        Args:
            text: Text string to embed
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass


class SentenceTransformerEmbedding(EmbeddingModel):
    """
    Embedding model using sentence-transformers library.
    
    This is a popular choice for semantic similarity tasks and works well
    for code and documentation embedding.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model.
        
        Args:
            model_name: Name of the pre-trained model to use.
                       Default is 'all-MiniLM-L6-v2' which is lightweight
                       and performs well for general-purpose embeddings.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install sentence-transformers"
            )
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim


class SimpleEmbedding(EmbeddingModel):
    """
    Simple TF-IDF based embedding for testing or lightweight use.
    
    This implementation doesn't require additional dependencies and can be
    used as a fallback when more sophisticated models are not available.
    """
    
    def __init__(self, max_features: int = 384):
        """
        Initialize TF-IDF based embedding.
        
        Args:
            max_features: Maximum number of features (embedding dimension)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self._embedding_dim = max_features
        self._is_fitted = False
    
    def fit(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer on a corpus.
        
        Must be called before embed() or embed_single().
        
        Args:
            texts: Corpus of texts to fit the vectorizer
        """
        self.vectorizer.fit(texts)
        self._is_fitted = True
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts using TF-IDF."""
        if not self._is_fitted:
            # Auto-fit on the provided texts
            self.fit(texts)
        return self.vectorizer.transform(texts).toarray()
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text using TF-IDF."""
        if not self._is_fitted:
            raise RuntimeError("SimpleEmbedding must be fitted before use. Call fit() first.")
        return self.vectorizer.transform([text]).toarray()[0]
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        return self._embedding_dim
