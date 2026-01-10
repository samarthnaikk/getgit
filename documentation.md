# GetGit Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Dependencies](#dependencies)
5. [Basic Usage](#basic-usage)
6. [Configuration](#configuration)
7. [Logging](#logging)
8. [API Reference](#api-reference)
9. [Contributing](#contributing)
10. [Roadmap](#roadmap)

---

## Project Overview

GetGit is a Python-based repository intelligence system that combines GitHub repository cloning, Retrieval-Augmented Generation (RAG), and Large Language Model (LLM) capabilities to provide intelligent, natural language question-answering over code repositories.

### Key Features

- **Automated Repository Cloning**: Clone and manage GitHub repositories locally
- **RAG-Based Analysis**: Semantic chunking and retrieval of repository content
- **LLM Integration**: Natural language response generation using Google Gemini
- **Unified Pipeline**: Single entry point (`core.py`) orchestrating all components
- **Web Interface**: Flask-based UI for repository exploration

### Use Cases

- Understanding unfamiliar codebases quickly
- Answering questions about project structure and functionality
- Extracting information from documentation and code
- Repository analysis and review
- Team collaboration and onboarding

---

## Architecture

GetGit follows a modular architecture with three main layers:

### 1. Repository Layer (`clone_repo.py`)

Handles GitHub repository cloning and local storage management.

**Key Function:**
```python
clone_repo(github_url, dest_folder='source_repo')
```

### 2. RAG Layer (`rag/` module)

Provides semantic search and context retrieval capabilities.

**Components:**
- **Chunker** (`chunker.py`): Splits repository files into semantic chunks
- **Embedder** (`embedder.py`): Creates vector embeddings (TF-IDF or Transformer-based)
- **Retriever** (`retriever.py`): Performs similarity-based chunk retrieval
- **Configuration** (`config.py`): Manages RAG settings and parameters

**Supported Chunk Types:**
- Code functions and classes
- Markdown sections
- Documentation blocks
- Configuration files
- Full file content

### 3. LLM Layer (`rag/llm_connector.py`)

Integrates with Large Language Models for natural language response generation.

**Provider Support:**
- Google Gemini (default: `gemini-2.0-flash-exp`)
- Extensible design for additional providers

### 4. Orchestration Layer (`core.py`)

Unified entry point that coordinates all components:

1. **Repository Initialization**: Clone or load repository
2. **RAG Setup**: Chunk, embed, and index repository content
3. **Query Processing**: Retrieve context and generate responses

### 5. Web Interface (`server.py`)

Flask-based web application providing a user-friendly interface for repository exploration.

---

## Installation

### Prerequisites

- Python 3.6 or higher
- pip package manager
- Git (for repository cloning)

### Installation Steps

1. **Clone the GetGit repository:**
   ```bash
   git clone https://github.com/samarthnaikk/getgit.git
   cd getgit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python core.py --help
   ```

---

## Dependencies

### Core Dependencies (Required)

- **GitPython**: Repository cloning and Git operations
- **numpy >= 1.20.0**: Numerical computations for embeddings
- **scikit-learn >= 0.24.0**: TF-IDF embeddings and similarity calculations
- **Flask >= 2.0.0**: Web interface

### Optional Dependencies

- **sentence-transformers >= 2.0.0**: Advanced semantic embeddings (recommended for better retrieval quality)
- **google-generativeai >= 0.3.0**: LLM-based response generation with Google Gemini
- **python-dotenv >= 0.19.0**: Environment variable management for API keys

### Installing Optional Dependencies

All dependencies (core + optional) are included in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For minimal installation (without LLM support):
```bash
pip install GitPython numpy>=1.20.0 scikit-learn>=0.24.0 Flask>=2.0.0
```

---

## Basic Usage

### Using the Core Module

The `core.py` module is the primary interface for GetGit functionality.

#### 1. Simple Query (Python API)

```python
from core import main

# Analyze a repository and answer a question
result = main(
    repo_url="https://github.com/username/repository.git",
    query="How do I install this project?",
    use_llm=False  # Set to True if GEMINI_API_KEY is configured
)

# Access results
print(result['response'])
for chunk in result['retrieved_chunks']:
    print(f"- {chunk['file_path']} (score: {chunk['score']:.3f})")
```

#### 2. Command-Line Usage

```bash
# Basic usage with default settings
python core.py https://github.com/username/repo.git "What is this project about?"

# With custom repository path
python core.py https://github.com/username/repo.git "How do I configure authentication?"
```

#### 3. Step-by-Step Control

For more granular control over the pipeline:

```python
from core import initialize_repository, setup_rag, answer_query

# Step 1: Clone/load repository
repo_path = initialize_repository(
    repo_url="https://github.com/username/repo.git",
    local_path="my_repo"
)

# Step 2: Setup RAG pipeline
retriever = setup_rag(repo_path)

# Step 3: Process multiple queries
queries = [
    "What are the main features?",
    "How do I run tests?",
    "What dependencies are required?"
]

for query in queries:
    result = answer_query(
        query=query,
        retriever=retriever,
        use_llm=False
    )
    print(f"Q: {query}")
    print(f"A: {result['retrieved_chunks'][0]['file_path']}\n")
```

### Using the RAG Module Directly

For custom RAG workflows:

```python
from rag import RepositoryChunker, SimpleEmbedding, Retriever, RAGConfig

# Initialize configuration
config = RAGConfig.default()

# Chunk repository
chunker = RepositoryChunker('/path/to/repo', repository_name='my-repo')
chunks = chunker.chunk_repository(config.chunking.file_patterns)

# Create and index retriever
embedding_model = SimpleEmbedding(max_features=384)
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

# Perform queries
results = retriever.retrieve("authentication setup", top_k=5)
for result in results:
    print(f"{result.chunk.file_path}: {result.score:.4f}")
```

### Example Script

A comprehensive usage example is provided in `example_core_usage.py`:

```bash
# Run interactive examples
python example_core_usage.py

# Run all examples non-interactively
python example_core_usage.py --all
```

---

## Configuration

### Environment Variables

GetGit uses environment variables for sensitive configuration:

#### LLM API Key (Optional)

Required for LLM-based response generation:

```bash
export GEMINI_API_KEY=your_api_key_here
```

Or create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

**Getting a Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Generate a new API key
4. Set it as an environment variable or in `.env`

### RAG Configuration

RAG behavior can be customized using `RAGConfig`:

```python
from rag import RAGConfig

# Use default configuration
config = RAGConfig.default()

# Use documentation-optimized configuration
config = RAGConfig.for_documentation()

# Custom configuration
config = RAGConfig(
    chunking=ChunkingConfig(
        file_patterns=['*.py', '*.md', '*.txt'],
        chunk_size=500,
        chunk_overlap=50
    ),
    embedding=EmbeddingConfig(
        model_type='sentence-transformer',
        model_name='all-MiniLM-L6-v2',
        embedding_dim=384
    )
)

# Use in core module
from core import main
result = main(repo_url="...", query="...", config=config)
```

### Repository Storage

By default, repositories are cloned to `source_repo/`:

```python
# Custom repository path
from core import initialize_repository

repo_path = initialize_repository(
    repo_url="https://github.com/user/repo.git",
    local_path="custom_path"
)
```

### Retrieval Parameters

Control number of context chunks retrieved:

```python
from core import answer_query

result = answer_query(
    query="How do I configure logging?",
    retriever=retriever,
    top_k=10,  # Retrieve top 10 chunks (default: 5)
    use_llm=True
)
```

---

## Logging

GetGit uses Python's standard `logging` module for comprehensive activity tracking.

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for unexpected situations
- **ERROR**: Error messages for failures

### Configuring Log Level

#### Via Core Module

```python
from core import main

result = main(
    repo_url="...",
    query="...",
    log_level="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
)
```

#### Programmatically

```python
from core import setup_logging

logger = setup_logging(level="DEBUG")
logger.debug("This is a debug message")
logger.info("This is an info message")
```

### Log Format

Logs follow this format:
```
YYYY-MM-DD HH:MM:SS - getgit.core - LEVEL - Message
```

Example:
```
2026-01-10 12:34:56 - getgit.core - INFO - Initializing repository from https://github.com/user/repo.git
2026-01-10 12:35:02 - getgit.core - INFO - Created 1247 chunks from repository
2026-01-10 12:35:08 - getgit.core - INFO - Successfully indexed 1247 chunks
```

### Typical Log Output

During a successful pipeline run:

1. **Repository Initialization**:
   ```
   INFO - Initializing repository from https://github.com/user/repo.git
   INFO - Repository successfully cloned to source_repo
   ```

2. **RAG Setup**:
   ```
   INFO - Setting up RAG pipeline for repository at source_repo
   INFO - Chunking repository content...
   INFO - Created 850 chunks from repository
   INFO - Using SimpleEmbedding (TF-IDF based)
   INFO - Successfully indexed 850 chunks
   ```

3. **Query Processing**:
   ```
   INFO - Processing query: 'How do I configure authentication?'
   INFO - Retrieving top 5 relevant chunks...
   INFO - Retrieved 5 relevant chunks
   INFO - Generating LLM response...
   INFO - LLM response generated successfully
   ```

---

## API Reference

### Core Module Functions

#### `main(repo_url, query, local_path='source_repo', use_llm=True, top_k=5, log_level='INFO', config=None)`

Orchestrates the full GetGit pipeline.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `query` (str): Natural language question about the repository
- `local_path` (str): Local path for repository storage (default: 'source_repo')
- `use_llm` (bool): Whether to generate LLM responses (default: True)
- `top_k` (int): Number of relevant chunks to retrieve (default: 5)
- `log_level` (str): Logging level - DEBUG, INFO, WARNING, ERROR (default: 'INFO')
- `config` (RAGConfig): Optional RAG configuration (default: None, uses default config)

**Returns:**
- Dictionary containing:
  - `query`: The original query
  - `retrieved_chunks`: List of retrieved chunk information
  - `context`: Combined context from retrieved chunks
  - `response`: Generated LLM response (if use_llm=True)
  - `error`: Error message if LLM generation fails

**Example:**
```python
result = main(
    repo_url="https://github.com/user/repo.git",
    query="What is this project about?",
    use_llm=True,
    log_level="INFO"
)
```

---

#### `initialize_repository(repo_url, local_path='source_repo')`

Clone or load a repository.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `local_path` (str): Local path for repository storage

**Returns:**
- str: Path to the cloned/loaded repository

**Example:**
```python
repo_path = initialize_repository(
    repo_url="https://github.com/user/repo.git",
    local_path="my_repo"
)
```

---

#### `setup_rag(repo_path, repository_name=None, config=None, use_sentence_transformer=False)`

Initialize RAG pipeline with chunking, embeddings, and retrieval.

**Parameters:**
- `repo_path` (str): Path to the repository
- `repository_name` (str): Optional repository name (default: basename of repo_path)
- `config` (RAGConfig): Optional RAG configuration (default: None, uses default config)
- `use_sentence_transformer` (bool): Use SentenceTransformer embeddings (default: False)

**Returns:**
- Retriever: Configured retriever instance with indexed chunks

**Example:**
```python
retriever = setup_rag(
    repo_path="source_repo",
    use_sentence_transformer=True
)
```

---

#### `answer_query(query, retriever, top_k=5, use_llm=True, api_key=None, model_name='gemini-2.0-flash-exp')`

Retrieve context and generate response for a query.

**Parameters:**
- `query` (str): Natural language question
- `retriever` (Retriever): Configured retriever instance
- `top_k` (int): Number of chunks to retrieve (default: 5)
- `use_llm` (bool): Whether to generate LLM response (default: True)
- `api_key` (str): Optional API key (default: None, reads from environment)
- `model_name` (str): LLM model name (default: 'gemini-2.0-flash-exp')

**Returns:**
- Dictionary with query results (same structure as `main()`)

**Example:**
```python
result = answer_query(
    query="How do I run tests?",
    retriever=retriever,
    top_k=3,
    use_llm=False
)
```

---

### RAG Module

#### `RepositoryChunker(repo_path, repository_name='')`

Chunks repository files into semantic units.

**Methods:**
- `chunk_repository(file_patterns)`: Chunk files matching patterns

**Example:**
```python
from rag import RepositoryChunker, RAGConfig

config = RAGConfig.default()
chunker = RepositoryChunker('/path/to/repo', repository_name='my-repo')
chunks = chunker.chunk_repository(config.chunking.file_patterns)
```

---

#### `Retriever(embedding_model)`

Performs similarity-based retrieval of chunks.

**Methods:**
- `index_chunks(chunks, batch_size=32)`: Index chunks for retrieval
- `retrieve(query, top_k=5)`: Retrieve top-k relevant chunks
- `save(path)`: Save retriever state to disk
- `load(path)`: Load retriever state from disk

**Example:**
```python
from rag import Retriever, SimpleEmbedding

embedding_model = SimpleEmbedding(max_features=384)
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

results = retriever.retrieve("authentication", top_k=5)
```

---

#### `generate_response(query, context_chunks, model_name='gemini-2.0-flash-exp', api_key=None)`

Generate LLM response using retrieved context.

**Parameters:**
- `query` (str): User query
- `context_chunks` (List[str]): Retrieved context chunks
- `model_name` (str): LLM model name
- `api_key` (str): API key (reads from environment if not provided)

**Returns:**
- str: Generated response

**Example:**
```python
from rag import generate_response

response = generate_response(
    query="How do I install this?",
    context_chunks=[chunk.content for chunk in results]
)
```

---

## Contributing

We welcome contributions to GetGit! Please follow these guidelines:

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and modular

### Development Workflow

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/your-username/getgit.git
   cd getgit
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes** and test thoroughly

5. **Commit with clear messages:**
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

### Testing

Before submitting a pull request:

1. Test your changes with multiple repositories
2. Verify both RAG-only and RAG+LLM workflows
3. Check for edge cases and error handling
4. Ensure logging is appropriate and informative

### Documentation

- Update README.md if you add user-facing features
- Update this documentation.md for technical changes
- Add docstrings and comments for complex logic
- Include usage examples for new functionality

### Areas for Contribution

- Additional LLM provider integrations (OpenAI, Anthropic, etc.)
- Enhanced chunking strategies for different file types
- Performance optimizations for large repositories
- Web interface improvements
- Test coverage and test infrastructure
- Documentation and examples

---

## Roadmap

GetGit is under active development. Planned features and improvements:

### Short-term (Next Release)

- **Enhanced Web Interface**
  - Interactive query interface
  - Repository management dashboard
  - Visual representation of retrieval results

- **Additional LLM Providers**
  - OpenAI GPT integration
  - Anthropic Claude integration
  - Local model support (Ollama, LLaMA)

- **Improved Chunking**
  - Language-specific code parsing
  - Smart function/class boundary detection
  - Configuration file semantic understanding

### Medium-term

- **Multi-Repository Analysis**
  - Track and query across multiple repositories simultaneously
  - Cross-repository search and comparison
  - Unified bookmarking and milestone tracking

- **Repository Review System**
  - Automated code quality analysis
  - Documentation completeness checking
  - Security vulnerability scanning
  - Best practices compliance reporting

- **Caching and Performance**
  - Persistent chunk and embedding storage
  - Incremental repository updates
  - Query result caching
  - Parallel processing for large repositories

### Long-term

- **Advanced QnA Capabilities**
  - Multi-turn conversational queries
  - Context-aware follow-up questions
  - Query history and session management
  - Personalized query responses

- **Team Collaboration Features**
  - Shared repository intelligence
  - Team annotations and bookmarks
  - Collaborative code review
  - Knowledge base building

- **Integration and Ecosystem**
  - GitHub App/Action integration
  - VS Code extension
  - Slack/Discord bot
  - API for third-party integrations

### Community-Driven

We encourage the community to suggest and contribute features. Please:

- Open issues for feature requests
- Participate in design discussions
- Submit pull requests for implementations
- Share feedback on existing features

---

## Support and Resources

- **Repository**: [https://github.com/samarthnaikk/getgit](https://github.com/samarthnaikk/getgit)
- **Issues**: [https://github.com/samarthnaikk/getgit/issues](https://github.com/samarthnaikk/getgit/issues)
- **License**: MIT License (see LICENSE file)

For questions, bug reports, or feature requests, please open an issue on GitHub.

---

*Last updated: January 2026*
