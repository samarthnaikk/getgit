# GetGit Technical Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Backend Flow](#backend-flow)
4. [RAG + LLM Overview](#rag--llm-overview)
5. [Checkpoints System](#checkpoints-system)
6. [UI Interaction Flow](#ui-interaction-flow)
7. [Setup and Run Instructions](#setup-and-run-instructions)
8. [Logging Behavior](#logging-behavior)
9. [API Reference](#api-reference)
10. [Configuration](#configuration)

---

## Project Overview

GetGit is a Python-based repository intelligence system that combines GitHub repository cloning, Retrieval-Augmented Generation (RAG), and Large Language Model (LLM) capabilities to provide intelligent, natural language question-answering over code repositories.

### Key Features

- **Automated Repository Cloning**: Clone and manage GitHub repositories locally
- **RAG-Based Analysis**: Semantic chunking and retrieval of repository content
- **LLM Integration**: Natural language response generation using Google Gemini
- **Checkpoint Validation**: Programmatic validation of repository requirements
- **Web Interface**: Flask-based UI for repository exploration
- **Checkpoint Management**: UI for adding and viewing validation checkpoints

### Use Cases

- Understanding unfamiliar codebases quickly
- Answering questions about project structure and functionality
- Extracting information from documentation and code
- Repository analysis and review
- Validating repository requirements for hackathons or project submissions
- Team collaboration and onboarding

---

## Architecture

GetGit follows a modular architecture with clear separation of concerns:

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                       Web Browser                            │
│                    (User Interface)                          │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    server.py (Flask)                         │
│  - Routes: /initialize, /ask, /checkpoints, etc.            │
│  - Session management                                        │
│  - Request/response handling                                 │
└────────────────────┬────────────────────────────────────────┘
                     │ Delegates to
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    core.py (Orchestration)                   │
│  - initialize_repository()                                   │
│  - setup_rag()                                              │
│  - answer_query()                                           │
│  - validate_checkpoints()                                   │
└────────┬───────────────────┬─────────────────┬──────────────┘
         │                   │                 │
         ▼                   ▼                 ▼
┌─────────────────┐  ┌──────────────┐  ┌─────────────────────┐
│  clone_repo.py  │  │   rag/       │  │  checkpoints.py     │
│  - Repository   │  │  - Chunker   │  │  - Load/validate    │
│    cloning      │  │  - Embedder  │  │  - Checkpoint mgmt  │
└─────────────────┘  │  - Retriever │  └─────────────────────┘
                     │  - LLM       │
                     └──────────────┘
```

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
- **LLM Connector** (`llm_connector.py`): Integrates with LLMs for response generation
- **Configuration** (`config.py`): Manages RAG settings and parameters

**Supported Chunk Types:**
- Code functions and classes
- Markdown sections
- Documentation blocks
- Configuration files
- Full file content

### 3. Checkpoints Layer (`checkpoints.py`)

Manages checkpoint-based validation of repositories.

**Key Functions:**
- `load_checkpoints()`: Load checkpoints from file
- `evaluate_checkpoint()`: Evaluate a single checkpoint
- `run_checkpoints()`: Run all checkpoints against repository
- `format_results_summary()`: Format results for display

### 4. Orchestration Layer (`core.py`)

Unified entry point that coordinates all components:

1. **Repository Initialization**: Clone or load repository
2. **RAG Setup**: Chunk, embed, and index repository content
3. **Query Processing**: Retrieve context and generate responses
4. **Checkpoint Validation**: Validate repository against requirements

### 5. Web Interface (`server.py`)

Flask-based web application providing a user-friendly interface.

**Routes:**
- `GET /` - Render home page
- `POST /initialize` - Initialize repository and RAG pipeline
- `POST /ask` - Answer questions about repository
- `POST /checkpoints` - Run checkpoint validation
- `GET /checkpoints/list` - List all checkpoints
- `POST /checkpoints/add` - Add new checkpoint
- `GET /status` - Get application status

---

## Backend Flow

### Server.py → Core.py Flow

```
User Request → server.py → core.py → Specialized Modules
```

#### 1. Repository Initialization Flow

```
POST /initialize
  ↓
server.py: initialize()
  ↓
core.py: initialize_repository(repo_url, local_path)
  ↓
clone_repo.py: clone_repo(repo_url, local_path)
  ↓
core.py: setup_rag(repo_path)
  ↓
rag/chunker.py: chunk_repository()
  ↓
rag/embedder.py: create embeddings
  ↓
rag/retriever.py: index_chunks()
  ↓
Return: Retriever instance with indexed chunks
```

#### 2. Question Answering Flow

```
POST /ask
  ↓
server.py: ask_question()
  ↓
core.py: answer_query(query, retriever, use_llm)
  ↓
rag/retriever.py: retrieve(query, top_k)
  ↓
[If use_llm=True]
  ↓
rag/llm_connector.py: generate_response(query, context)
  ↓
Return: {query, retrieved_chunks, context, response, error}
```

#### 3. Checkpoint Validation Flow

```
POST /checkpoints
  ↓
server.py: run_checkpoints()
  ↓
core.py: validate_checkpoints(repo_url, checkpoints_file, use_llm)
  ↓
checkpoints.py: load_checkpoints(file)
  ↓
checkpoints.py: run_checkpoints(checkpoints, repo_path, retriever)
  ↓
[For each checkpoint]
  ↓
checkpoints.py: evaluate_checkpoint(checkpoint, retriever, use_llm)
  ↓
Return: {checkpoints, results, summary, statistics}
```

---

## RAG + LLM Overview

### Retrieval-Augmented Generation (RAG)

RAG combines information retrieval with text generation to provide contextually accurate responses.

**How It Works:**

1. **Indexing Phase** (Setup):
   - Repository files are chunked into semantic units
   - Each chunk is converted to a vector embedding
   - Embeddings are indexed for fast similarity search

2. **Retrieval Phase** (Query):
   - User query is converted to embedding
   - Similar chunks are retrieved using cosine similarity
   - Top-k most relevant chunks are selected

3. **Generation Phase** (Optional, if LLM enabled):
   - Retrieved chunks provide context
   - Context + query sent to LLM
   - LLM generates coherent, contextual response

### LLM Integration

GetGit uses Google Gemini for natural language response generation.

**Features:**
- Provider-agnostic design (easy to add new LLM providers)
- Environment-based API key management
- Error handling and fallback to context-only responses
- Configurable model selection

**Configuration:**
```bash
export GEMINI_API_KEY=your_api_key_here
```

---

## Checkpoints System

The checkpoints system enables programmatic validation of repository requirements.

### How Checkpoints Work

1. **Definition**: Checkpoints are stored in `checkpoints.txt`, one per line
2. **Loading**: System reads and parses checkpoint file
3. **Evaluation**: Each checkpoint is evaluated against the repository
4. **Reporting**: Results include pass/fail status, explanation, and evidence

### Checkpoint Types

1. **File Existence Checks**: Simple file/directory existence validation
   - Example: "Check if the repository has README.md"

2. **Semantic Checks**: Complex requirements using RAG retrieval
   - Example: "Check if RAG model is implemented"

3. **LLM-Enhanced Checks**: Uses LLM reasoning for complex validation
   - Example: "Check if proper error handling is implemented"

### Checkpoints File Format

```
# Comments start with #
1. Check if the repository has README.md
2. Check if RAG model is implemented
3. Check if logging is configured
Check if requirements.txt exists  # Numbering is optional
```

### Managing Checkpoints via UI

The web interface provides checkpoint management:
- **View Checkpoints**: Load and display all checkpoints from file
- **Add Checkpoint**: Add new checkpoints via UI
- **Persistence**: All checkpoints saved to `checkpoints.txt`
- **Server Restart**: Checkpoints persist across server restarts

---

## UI Interaction Flow

### User Journey

1. **Initialize Repository**
   - User enters GitHub repository URL
   - Clicks "Initialize Repository"
   - Backend clones repository and indexes content
   - UI displays success message and chunk count

2. **Manage Checkpoints**
   - User can add new checkpoint requirements
   - User can view existing checkpoints
   - Checkpoints saved to `checkpoints.txt`
   - Available for validation

3. **Ask Questions**
   - User enters natural language question
   - Optionally enables LLM for enhanced responses
   - Backend retrieves relevant code chunks
   - UI displays answer and source chunks

4. **Run Validation**
   - User triggers checkpoint validation
   - Backend evaluates all checkpoints
   - UI displays pass/fail results with explanations

### UI Components

- **Status Messages**: Success, error, and info notifications
- **Loading Indicators**: Spinner during processing
- **Result Boxes**: Formatted display of results
- **Checkpoint List**: Scrollable list of checkpoints
- **Forms**: Input fields for URLs, questions, checkpoints

---

## Setup and Run Instructions

### Prerequisites

- Python 3.6 or higher
- pip package manager
- Git (for repository cloning)

### Installation

1. **Clone GetGit repository:**
   ```bash
   git clone https://github.com/samarthnaikk/getgit.git
   cd getgit
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional):**
   ```bash
   # For LLM-powered responses
   export GEMINI_API_KEY=your_api_key_here
   
   # For production deployment

   ```

### Running the Application

**Development Mode:**
```bash
FLASK_ENV=development python server.py
```

**Production Mode:**
```bash
python server.py
```

The server will start on `http://0.0.0.0:5000`

### Accessing the UI

Open your web browser and navigate to:
```
http://localhost:5000
```

---

## Logging Behavior

GetGit uses Python's standard `logging` module for comprehensive activity tracking.

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (default)
- **WARNING**: Warning messages for unexpected situations
- **ERROR**: Error messages for failures

### Log Format

```
YYYY-MM-DD HH:MM:SS - getgit.MODULE - LEVEL - Message
```

Example:
```
2026-01-10 12:34:56 - getgit.core - INFO - Initializing repository from https://github.com/user/repo.git
2026-01-10 12:35:02 - getgit.core - INFO - Created 1247 chunks from repository
2026-01-10 12:35:08 - getgit.server - INFO - Repository initialization completed successfully
```

### Server Logs

Server logs include:
- Request processing
- Route handling
- Success/failure of operations
- Error stack traces (when errors occur)

### Core Module Logs

Core module logs include:
- Repository initialization progress
- RAG pipeline setup stages
- Query processing steps
- Checkpoint validation progress

### Configuring Log Level

**Via Environment:**
```bash
# Not directly supported, modify code or use Python logging config
```

**In Code:**
```python
from core import setup_logging
logger = setup_logging(level="DEBUG")
```

---

## API Reference

### Core Module Functions

#### `initialize_repository(repo_url, local_path='source_repo')`

Clone or load a repository and prepare it for analysis.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `local_path` (str): Local path for repository storage

**Returns:** str - Path to the cloned/loaded repository

**Example:**
```python
from core import initialize_repository
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
- `repository_name` (str, optional): Repository name
- `config` (RAGConfig, optional): RAG configuration
- `use_sentence_transformer` (bool): Use transformer embeddings

**Returns:** Retriever - Configured retriever instance

**Example:**
```python
from core import setup_rag
retriever = setup_rag(repo_path="source_repo")
```

---

#### `answer_query(query, retriever, top_k=5, use_llm=True, api_key=None, model_name='gemini-2.0-flash-exp')`

Retrieve context and generate response for a query.

**Parameters:**
- `query` (str): Natural language question
- `retriever` (Retriever): Configured retriever instance
- `top_k` (int): Number of chunks to retrieve
- `use_llm` (bool): Whether to generate LLM response
- `api_key` (str, optional): API key for LLM
- `model_name` (str): LLM model name

**Returns:** dict - Query results with response and context

**Example:**
```python
from core import answer_query
result = answer_query(
    query="How do I run tests?",
    retriever=retriever,
    top_k=5,
    use_llm=True
)
```

---

#### `validate_checkpoints(repo_url, checkpoints_file='checkpoints.txt', local_path='source_repo', use_llm=True, log_level='INFO', config=None, stop_on_failure=False)`

Validate repository against checkpoints defined in a text file.

**Parameters:**
- `repo_url` (str): GitHub repository URL
- `checkpoints_file` (str): Path to checkpoints file
- `local_path` (str): Local repository storage path
- `use_llm` (bool): Use LLM for evaluation
- `log_level` (str): Logging level
- `config` (RAGConfig, optional): RAG configuration
- `stop_on_failure` (bool): Stop on first failure

**Returns:** dict - Validation results with statistics

**Example:**
```python
from core import validate_checkpoints
result = validate_checkpoints(
    repo_url="https://github.com/user/repo.git",
    checkpoints_file="checkpoints.txt",
    use_llm=True
)
print(result['summary'])
```

---

### Flask API Endpoints

#### `POST /initialize`

Initialize repository and setup RAG pipeline.

**Request Body:**
```json
{
  "repo_url": "https://github.com/user/repo.git"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Repository initialized successfully with 850 chunks",
  "repo_path": "source_repo",
  "chunks_count": 850
}
```

---

#### `POST /ask`

Answer questions about the repository.

**Request Body:**
```json
{
  "query": "What is this project about?",
  "use_llm": true
}
```

**Response:**
```json
{
  "success": true,
  "query": "What is this project about?",
  "response": "This project is a repository intelligence system...",
  "retrieved_chunks": [...],
  "context": "...",
  "error": null
}
```

---

#### `POST /checkpoints`

Run checkpoint validation.

**Request Body:**
```json
{
  "checkpoints_file": "checkpoints.txt",
  "use_llm": true
}
```

**Response:**
```json
{
  "success": true,
  "checkpoints": ["Check if README exists", ...],
  "results": [{
    "checkpoint": "Check if README exists",
    "passed": true,
    "explanation": "...",
    "evidence": "...",
    "score": 1.0
  }],
  "summary": "...",
  "passed_count": 4,
  "total_count": 5,
  "pass_rate": 80.0
}
```

---

#### `GET /checkpoints/list`

List all checkpoints from checkpoints.txt.

**Response:**
```json
{
  "success": true,
  "checkpoints": [
    "Check if the repository has README.md",
    "Check if RAG model is implemented"
  ]
}
```

---

#### `POST /checkpoints/add`

Add a new checkpoint to checkpoints.txt.

**Request Body:**
```json
{
  "checkpoint": "Check if tests are present"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Checkpoint added successfully",
  "checkpoints": [...]
}
```

---

#### `GET /status`

Get current application status.

**Response:**
```json
{
  "initialized": true,
  "repo_url": "https://github.com/user/repo.git",
  "chunks_count": 850
}
```

---

## Configuration

### Environment Variables

- **GEMINI_API_KEY**: API key for Google Gemini LLM (optional)

- **FLASK_ENV**: Set to `development` for debug mode

### RAG Configuration

```python
from rag import RAGConfig

# Use default configuration
config = RAGConfig.default()

# Use documentation-optimized configuration
config = RAGConfig.for_documentation()

# Custom configuration
from rag import ChunkingConfig, EmbeddingConfig

config = RAGConfig(
    chunking=ChunkingConfig(
        file_patterns=['*.py', '*.md'],
        chunk_size=500,
        chunk_overlap=50
    ),
    embedding=EmbeddingConfig(
        model_type='sentence-transformer',
        embedding_dim=384
    )
)
```

### Repository Storage

By default, repositories are cloned to `source_repo/`. This can be customized via the `local_path` parameter.

---

*Last updated: January 2026*
   ```bash
   git clone https://github.com/samarthnaikk/getgit.git
