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

For a complete example, run:
```bash
python example_rag_usage.py
```

## Requirements

### Core Dependencies
- Python 3.6 or higher
- GitPython library
- numpy >= 1.20.0
- scikit-learn >= 0.24.0

### Optional Dependencies
- sentence-transformers >= 2.0.0 (for advanced semantic embeddings)

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
