"""
RAG (Retrieval-Augmented Generation) module for GetGit.

This module provides chunking and retrieval capabilities for repository analysis,
enabling semantic search and context extraction from codebases, documentation,
and commit history.
"""

from .chunker import RepositoryChunker, Chunk, ChunkType
from .embedder import EmbeddingModel, SentenceTransformerEmbedding, SimpleEmbedding
from .retriever import VectorStore, Retriever, InMemoryVectorStore, RetrievalResult
from .config import RAGConfig, ChunkingConfig, EmbeddingConfig, RetrievalConfig

__all__ = [
    'RepositoryChunker',
    'Chunk',
    'ChunkType',
    'EmbeddingModel',
    'SentenceTransformerEmbedding',
    'SimpleEmbedding',
    'VectorStore',
    'InMemoryVectorStore',
    'Retriever',
    'RetrievalResult',
    'RAGConfig',
    'ChunkingConfig',
    'EmbeddingConfig',
    'RetrievalConfig',
]
