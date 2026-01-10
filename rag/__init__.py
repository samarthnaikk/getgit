"""
RAG (Retrieval-Augmented Generation) module for GetGit.

This module provides chunking, retrieval, and generation capabilities for repository analysis,
enabling semantic search, context extraction, and LLM-based response generation from codebases,
documentation, and commit history.
"""

from .chunker import RepositoryChunker, Chunk, ChunkType
from .embedder import EmbeddingModel, SentenceTransformerEmbedding, SimpleEmbedding
from .retriever import VectorStore, Retriever, InMemoryVectorStore, RetrievalResult
from .config import RAGConfig, ChunkingConfig, EmbeddingConfig, RetrievalConfig
from .llm_connector import build_prompt, query_llm, generate_response

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
    'build_prompt',
    'query_llm',
    'generate_response',
]
