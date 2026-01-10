"""
Configuration management for RAG system.

Provides default configurations and allows customization of chunking,
embedding, and retrieval parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    
    # File patterns to include
    file_patterns: List[str] = field(default_factory=lambda: [
        '*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml'
    ])
    
    # Generic file chunking parameters
    generic_chunk_size: int = 50  # lines per chunk
    generic_overlap: int = 10     # lines of overlap
    
    # Exclude patterns (directories and files to skip)
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '__pycache__', 'node_modules', '.git', '*.pyc', '.DS_Store'
    ])


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    
    # Model type: 'sentence-transformer' or 'simple'
    model_type: str = 'simple'  # Default to simple to avoid external dependencies
    
    # Model name (for sentence-transformer)
    model_name: str = 'all-MiniLM-L6-v2'
    
    # Embedding dimension (for simple model)
    embedding_dim: int = 384
    
    # Batch size for embedding generation
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    
    # Default number of results to return
    default_top_k: int = 5
    
    # Vector store type: 'in-memory' (more can be added later)
    vector_store_type: str = 'in-memory'
    
    # Cache directory for storing vector indices
    cache_dir: str = '.rag_cache'


@dataclass
class RAGConfig:
    """Main RAG configuration combining all sub-configs."""
    
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    @classmethod
    def default(cls) -> 'RAGConfig':
        """Return default configuration."""
        return cls()
    
    @classmethod
    def for_large_repos(cls) -> 'RAGConfig':
        """Return configuration optimized for large repositories."""
        config = cls()
        config.chunking.generic_chunk_size = 100
        config.embedding.batch_size = 64
        return config
    
    @classmethod
    def for_code_only(cls) -> 'RAGConfig':
        """Return configuration for code-only analysis."""
        config = cls()
        config.chunking.file_patterns = ['*.py', '*.js', '*.java', '*.cpp', '*.c', '*.h']
        return config
    
    @classmethod
    def for_documentation(cls) -> 'RAGConfig':
        """Return configuration for documentation-focused analysis."""
        config = cls()
        config.chunking.file_patterns = ['*.md', '*.rst', '*.txt']
        return config
