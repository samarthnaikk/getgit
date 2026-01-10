# GetGit

## Overview

GetGit is a Python command-line tool designed to analyze GitHub repositories and provide intelligent repository insights through natural language interaction. The tool serves as a repository intelligence layer that leverages GitHub's REST API combined with advanced query pattern matching to enable users to explore and understand repositories without the need to manually browse through files, documentation, or metadata.

## About the Project

GetGit addresses the growing complexity of managing and understanding multiple GitHub repositories, particularly in collaborative environments such as hackathons, team projects, and code review workflows. By providing automated analysis and natural language query capabilities, GetGit streamlines the process of extracting meaningful information from repository structures, commit histories, and codebase patterns.

The tool is built with Python and utilizes GitPython for repository operations, providing a lightweight yet powerful solution for repository intelligence gathering.

## Current Functionality

The current implementation of GetGit provides:

- **Repository Cloning**: Automated cloning of GitHub repositories to a local destination folder with cleanup of existing directories
- **GitHub API Integration**: Direct integration with GitHub's REST API for retrieving repository metadata
- **Query Pattern Matching**: Pattern-based analysis system for understanding and responding to repository-related queries
- **RAG-Based Chunking and Retrieval**: Advanced semantic search and context extraction for repository analysis
  - Intelligent chunking of source code, documentation, and configuration files
  - Semantic embedding and vector-based retrieval
  - Natural language queries over repository content
  - Support for multiple file types and programming languages
- **LLM-Based Response Generation**: Integration with Large Language Models for natural language responses
  - Google Gemini integration for generating contextual answers
  - Combines retrieved repository context with LLM capabilities
  - Provider-agnostic design for easy extension to other LLMs
  - Environment-based API key management
- **Checkpoint-Based Validation System**: Automated validation of repository requirements
  - Text-based checkpoint definitions for programmatic validation
  - Support for file existence checks and complex semantic requirements
  - Pass/fail results with detailed explanations and evidence
  - Integration with RAG and LLM for intelligent evaluation
  - Ideal for hackathon submissions, project evaluations, and CI-style checks

## Planned Features

GetGit is under active development with several key features planned for future releases:

### Multi-Repository Milestone Tracking

This feature will enable users to track bookmarks and milestones across multiple repositories simultaneously. This is particularly useful in hackathon scenarios where teams work on the same project distributed across multiple repositories, or in organizations managing microservices architectures. The feature will provide:

- Unified view of milestones across selected repositories
- Bookmark synchronization for important commits or branches
- Progress tracking for multi-repository projects
- Timeline visualization of development activities

### Repository Review System

An automated repository review system that will analyze codebases and provide comprehensive feedback on:

- Code quality metrics and standards compliance
- Repository structure and organization
- Documentation completeness and accuracy
- Best practices adherence
- Security considerations and potential vulnerabilities
- Dependency management and update recommendations

This feature aims to facilitate peer review processes and maintain code quality standards across projects.

### Interactive Repository QnA

A natural language question-answering system that will allow users to interact with repositories through conversational queries. This feature will enable:

- Natural language queries about repository contents, structure, and history
- Contextual understanding of codebase relationships
- Automated generation of responses based on repository analysis
- Documentation extraction and summarization
- Commit history analysis and explanation
- Contributor activity and statistics insights

## Installation

To install GetGit, ensure you have Python 3.6 or higher installed on your system.

1. Clone the repository:
```bash
git clone https://github.com/samarthnaikk/getgit.git
cd getgit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Repository Cloning

The tool provides repository cloning functionality through the `clone_repo` module:

```python
from clone_repo import clone_repo

# Clone a GitHub repository
clone_repo('https://github.com/username/repository.git', 'destination_folder')
```

### RAG-Based Repository Analysis

The RAG (Retrieval-Augmented Generation) module enables semantic search and intelligent context extraction from repositories:

```python
from rag import RepositoryChunker, SimpleEmbedding, Retriever, RAGConfig

# Initialize configuration
config = RAGConfig.default()

# Chunk the repository
chunker = RepositoryChunker('path/to/repo', repository_name='my-repo')
chunks = chunker.chunk_repository(config.chunking.file_patterns)

# Create retriever and index chunks
embedding_model = SimpleEmbedding(max_features=384)
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

# Query the repository with natural language
results = retriever.retrieve("How do I configure authentication?", top_k=5)

# Display results
for result in results:
    print(f"[{result.rank}] {result.chunk.file_path} (score: {result.score:.4f})")
```

For complete examples, see `example_core_usage.py` or refer to `documentation.md` for detailed usage.

### LLM-Based Response Generation

The LLM connector module enables generating natural language responses by combining retrieved repository context with Large Language Models:

```python
from rag import RepositoryChunker, SimpleEmbedding, Retriever, generate_response, RAGConfig

# Initialize and index repository (as shown above)
config = RAGConfig.default()
chunker = RepositoryChunker('path/to/repo', repository_name='my-repo')
chunks = chunker.chunk_repository(config.chunking.file_patterns)
embedding_model = SimpleEmbedding(max_features=384)
retriever = Retriever(embedding_model)
retriever.index_chunks(chunks)

# Retrieve relevant context
query = "How do I configure authentication?"
results = retriever.retrieve(query, top_k=5)
context_chunks = [result.chunk.content for result in results]

# Generate LLM response using retrieved context
response = generate_response(query, context_chunks)
print(response)
```

#### Setting up the LLM API Key

The LLM connector uses Google Gemini by default. To use it, you need to set up your API key:

1. Get a Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```
3. Or export it as an environment variable:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

For complete examples with LLM integration, see `example_core_usage.py` or refer to `documentation.md` for detailed usage.

### Checkpoint-Based Validation

The checkpoint validation system enables programmatic validation of repository requirements using text-based checkpoints. This is useful for hackathon submissions, project evaluations, and CI-style checks.

#### Creating a Checkpoints File

Create a `checkpoints.txt` file with one requirement per line:

```
1. Check if the repository has README.md
2. Check if RAG model is implemented
3. Check if logging is configured
4. Check if requirements.txt exists
5. Check if the project uses Flask for web interface
```

Lines starting with `#` are treated as comments and ignored. Numbering is optional and will be stripped automatically.

#### Running Checkpoint Validation

```python
from core import validate_checkpoints

# Run checkpoint validation
result = validate_checkpoints(
    repo_url='https://github.com/username/repository.git',
    checkpoints_file='checkpoints.txt',
    use_llm=False,  # Set to True for enhanced evaluation with LLM
    log_level='INFO'
)

# Display results
print(result['summary'])
print(f"\nPassed: {result['passed_count']}/{result['total_count']}")
print(f"Pass Rate: {result['pass_rate']:.1f}%")
```

#### How It Works

The checkpoint validation system:

1. **Deterministic Checks**: Simple file existence checks are handled deterministically by scanning the repository
2. **RAG Retrieval**: Complex requirements use semantic search to find relevant code and documentation
3. **LLM Evaluation** (optional): When enabled, uses LLM reasoning to interpret complex requirements and provide detailed explanations

#### Example Output

```
[PASS] Check if the repository has README.md
  File 'README.md' found in repository
  Evidence: README.md
  Confidence: 1.00

[PASS] Check if RAG model is implemented
  Repository contains RAG implementation with retriever and embedding components
  Evidence: rag/retriever.py, rag/embedder.py, core.py
  Confidence: 0.89

[FAIL] Check if logging is configured
  No logging configuration found in repository
  Evidence: No relevant files detected
  Confidence: 0.12

======================================================================
SUMMARY
Total Checkpoints: 3
Passed: 2
Failed: 1
Pass Rate: 66.7%
```

For more examples, see `example_checkpoint_usage.py`.

## Requirements

### Core Dependencies
- Python 3.6 or higher
- GitPython library
- numpy >= 1.20.0
- scikit-learn >= 0.24.0

### Optional Dependencies
- sentence-transformers >= 2.0.0 (for advanced semantic embeddings)
- google-generativeai >= 0.3.0 (for LLM-based response generation)
- python-dotenv >= 0.19.0 (for environment variable management)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions to GetGit are welcome. When contributing, please:

- Follow Python PEP 8 style guidelines
- Provide clear commit messages
- Document new features and functionality
- Test changes thoroughly before submitting pull requests

## License

This project is licensed under the MIT License. See the LICENSE file for complete details.

## Project Status

GetGit is currently in active development. The core repository analysis functionality is operational, with several major features planned for upcoming releases. Community feedback and contributions are encouraged to help shape the direction of the project.
