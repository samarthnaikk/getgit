# RAG (Retrieval-Augmented Generation) Module

## Overview

The RAG module provides intelligent chunking and semantic retrieval capabilities for repository analysis in GetGit. It enables natural language queries over codebases, documentation, and configuration files by breaking them into semantically meaningful chunks, embedding them into vector space, and retrieving relevant context based on similarity.

## Features

- **Multi-Strategy Chunking**: Tailored chunking strategies for different file types
  - Python: Function-level and class-level chunking
  - Markdown: Section-based chunking by headers
  - Configuration files: Whole-file chunking
  - Generic files: Fixed-size chunking with overlap

- **Pluggable Embedding Models**: Support for multiple embedding approaches
  - `SimpleEmbedding`: TF-IDF based (no external dependencies)
  - `SentenceTransformerEmbedding`: State-of-the-art semantic embeddings (requires sentence-transformers)

- **Flexible Vector Storage**: Abstraction layer for different storage backends
  - `InMemoryVectorStore`: Fast in-memory storage with cosine similarity
  - Extensible to support FAISS, Pinecone, or other vector databases

- **Natural Language Retrieval**: Query repositories using natural language
  - Semantic search across all repository content
  - Ranked results with similarity scores
  - Type-based filtering (e.g., only functions, only documentation)

## Architecture

### Core Components

```
rag/
├── __init__.py          # Main module exports
├── chunker.py           # Chunking strategies for different file types
├── embedder.py          # Embedding model abstractions
├── retriever.py         # Vector storage and retrieval system
└── config.py            # Configuration management
```

### Data Flow

```
Repository Files → Chunker → Chunks → Embedder → Vectors → VectorStore
                                                                ↓
User Query → Embedder → Query Vector → Retriever → Ranked Results
```

## Quick Start

### Basic Usage

```python
from rag import (
    RepositoryChunker,
    SimpleEmbedding,
    Retriever,
    RAGConfig
)

# 1. Initialize configuration
config = RAGConfig.default()

# 2. Chunk the repository
chunker = RepositoryChunker('path/to/repo', repository_name='my-repo')
chunks = chunker.chunk_repository(config.chunking.file_patterns)

# 3. Create embeddings and retriever
embedding_model = SimpleEmbedding(max_features=384)
retriever = Retriever(embedding_model)

# 4. Index chunks
retriever.index_chunks(chunks)

# 5. Query the repository
results = retriever.retrieve("How do I configure authentication?", top_k=5)

# 6. Display results
for result in results:
    print(f"[{result.rank}] {result.chunk.file_path} (score: {result.score:.4f})")
    print(f"    {result.chunk.content[:100]}...")
```

### Using Advanced Embeddings

For better semantic understanding, use sentence-transformers:

```python
from rag import SentenceTransformerEmbedding

# Requires: pip install sentence-transformers
embedding_model = SentenceTransformerEmbedding('all-MiniLM-L6-v2')
retriever = Retriever(embedding_model)
```

### Saving and Loading Retriever State

```python
# Save indexed data
retriever.save('.rag_cache/my_repo.pkl')

# Load later
retriever = Retriever(embedding_model)
retriever.load('.rag_cache/my_repo.pkl')
```

## Configuration

### Predefined Configurations

```python
from rag import RAGConfig

# Default configuration (all supported file types)
config = RAGConfig.default()

# Optimized for large repositories
config = RAGConfig.for_large_repos()

# Code-only analysis
config = RAGConfig.for_code_only()

# Documentation-focused
config = RAGConfig.for_documentation()
```

### Custom Configuration

```python
from rag import RAGConfig, ChunkingConfig, EmbeddingConfig

config = RAGConfig()

# Customize chunking
config.chunking.file_patterns = ['*.py', '*.js']
config.chunking.generic_chunk_size = 100

# Customize embedding
config.embedding.model_type = 'sentence-transformer'
config.embedding.model_name = 'all-mpnet-base-v2'
config.embedding.batch_size = 64
```

## API Reference

### RepositoryChunker

**`__init__(repository_path: str, repository_name: str = "")`**

Initialize the chunker with a repository path.

**`chunk_repository(file_patterns: Optional[List[str]] = None) -> List[Chunk]`**

Chunk entire repository based on file patterns.

**`chunk_file(file_path: str, relative_path: str) -> List[Chunk]`**

Chunk a single file based on its type.

### Chunk

**Dataclass Attributes:**
- `content`: The actual text content
- `chunk_type`: Type of chunk (ChunkType enum)
- `file_path`: Relative path to the file
- `start_line`: Starting line number (1-indexed)
- `end_line`: Ending line number (1-indexed)
- `metadata`: Additional metadata (dict)
- `repository`: Repository identifier

### EmbeddingModel (Abstract Base Class)

**`embed(texts: List[str]) -> np.ndarray`**

Embed multiple text strings.

**`embed_single(text: str) -> np.ndarray`**

Embed a single text string.

**`embedding_dim: int`**

Property returning the dimensionality of embeddings.

### Retriever

**`__init__(embedding_model, vector_store: Optional[VectorStore] = None)`**

Initialize retriever with an embedding model and optional vector store.

**`index_chunks(chunks: List[Chunk], batch_size: int = 32)`**

Index chunks for retrieval.

**`retrieve(query: str, top_k: int = 5, filter_type: Optional[str] = None) -> List[RetrievalResult]`**

Retrieve relevant chunks for a natural language query.

**`save(filepath: str)`**

Save the retriever state to disk.

**`load(filepath: str)`**

Load the retriever state from disk.

### RetrievalResult

**Dataclass Attributes:**
- `chunk`: The retrieved Chunk object
- `score`: Similarity score (0-1, higher is better)
- `rank`: Rank in results (1-indexed)

## Chunk Types

The system recognizes the following chunk types:

- `CODE_FUNCTION`: Python function definitions
- `CODE_CLASS`: Python class definitions
- `CODE_METHOD`: Methods within classes
- `MARKDOWN_SECTION`: Markdown sections by header
- `DOCUMENTATION`: Documentation files
- `CONFIGURATION`: Config files (JSON, YAML)
- `COMMIT_MESSAGE`: Git commit messages (future)
- `GENERIC`: Generic text chunks

## Advanced Usage

### Type-Filtered Retrieval

```python
# Only retrieve code functions
results = retriever.retrieve(
    "authentication function",
    top_k=5,
    filter_type='code_function'
)
```

### Custom Embedding Model

Implement the `EmbeddingModel` interface:

```python
from rag import EmbeddingModel
import numpy as np

class MyCustomEmbedding(EmbeddingModel):
    def embed(self, texts: List[str]) -> np.ndarray:
        # Your implementation
        pass
    
    def embed_single(self, text: str) -> np.ndarray:
        # Your implementation
        pass
    
    @property
    def embedding_dim(self) -> int:
        return 768  # Your dimension
```

### Custom Vector Store

Implement the `VectorStore` interface:

```python
from rag import VectorStore

class MyVectorStore(VectorStore):
    def add_chunks(self, chunks, embeddings):
        # Your implementation
        pass
    
    def search(self, query_embedding, top_k=5):
        # Your implementation
        pass
    
    # ... implement other methods
```

## Performance Considerations

### Memory Usage

- **SimpleEmbedding**: ~50MB for typical repositories
- **SentenceTransformer**: ~100-400MB depending on model
- **InMemoryVectorStore**: ~1KB per chunk

### Processing Speed

- **Chunking**: ~1000 files/second
- **Embedding (Simple)**: ~100 chunks/second
- **Embedding (SentenceTransformer)**: ~50-200 chunks/second (GPU-dependent)
- **Retrieval**: <10ms for repositories with <10k chunks

### Optimization Tips

1. Use batch processing for large repositories
2. Cache embeddings on disk to avoid recomputation
3. For production use, consider FAISS for vector storage
4. Use GPU-accelerated embedding models when available

## Integration Examples

### With Repository QnA

```python
def answer_question(repo_path: str, question: str) -> str:
    """Answer questions about a repository."""
    chunker = RepositoryChunker(repo_path)
    chunks = chunker.chunk_repository()
    
    retriever = Retriever(SimpleEmbedding())
    retriever.index_chunks(chunks)
    
    results = retriever.retrieve(question, top_k=3)
    
    # Format context for LLM
    context = "\n\n".join([
        f"From {r.chunk.file_path}:\n{r.chunk.content}"
        for r in results
    ])
    
    return f"Based on:\n{context}\n\nAnswer: [Use LLM here]"
```

### With Repository Review

```python
def review_repository(repo_path: str) -> dict:
    """Generate repository review insights."""
    chunker = RepositoryChunker(repo_path)
    chunks = chunker.chunk_repository()
    
    # Analyze by type
    code_chunks = [c for c in chunks if 'code' in c.chunk_type.value]
    doc_chunks = [c for c in chunks if c.chunk_type.value == 'markdown_section']
    
    return {
        'total_chunks': len(chunks),
        'code_functions': len([c for c in code_chunks if c.chunk_type.value == 'code_function']),
        'documentation_sections': len(doc_chunks),
        # ... more analysis
    }
```

## Limitations and Future Work

### Current Limitations

- Python is the only fully supported programming language for code chunking
- Commit history chunking not yet implemented
- No built-in support for distributed vector stores
- Limited multi-repository analysis

### Planned Enhancements

- [ ] Support for more programming languages (JavaScript, Java, Go, etc.)
- [ ] Git commit and history analysis
- [ ] FAISS integration for large-scale retrieval
- [ ] Multi-repository cross-referencing
- [ ] Incremental indexing for repository updates
- [ ] Query expansion and semantic refinement
- [ ] Fine-tuned code embedding models

## Dependencies

### Required
- `numpy>=1.20.0`
- `scikit-learn>=0.24.0`

### Optional
- `sentence-transformers>=2.0.0` (for advanced embeddings)
- `faiss-cpu` or `faiss-gpu` (for large-scale vector search)

## License

This module is part of GetGit and is licensed under the MIT License.

## Contributing

Contributions are welcome! Areas of particular interest:

- Additional programming language support
- Alternative embedding models
- Vector store integrations
- Performance optimizations
- Test coverage improvements

Please follow the project's contribution guidelines when submitting changes.
