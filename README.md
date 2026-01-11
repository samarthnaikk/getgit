# GetGit

GetGit is a repository intelligence system that helps you analyze and understand GitHub repositories using natural language. It combines Retrieval-Augmented Generation (RAG) with Large Language Models to provide intelligent answers about any codebase.

## What GetGit Does

- **Repository Analysis**: Automatically indexes and analyzes GitHub repositories
- **Natural Language Queries**: Ask questions about code in plain English
- **Checkpoint Validation**: Programmatically validate repository requirements
- **Intelligent Search**: Semantic search across code, documentation, and configuration files

## Features

- **RAG-Based Analysis**: Semantic chunking and retrieval of repository content
- **Local LLM Support**: Integrated Hugging Face model (Qwen/Qwen2.5-Coder-7B) optimized for code intelligence
- **Automatic Fallback**: Falls back to Google Gemini if local model is unavailable
- **Repository Persistence**: Automatically tracks and validates indexed repositories
- **Smart Cleanup**: Detects repository changes and re-indexes automatically
- **Checkpoint System**: Automated validation of repository requirements
- **Web Interface**: User-friendly interface for repository exploration
- **Offline Capable**: Fully functional offline after initial model download

## How to Run

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t getgit .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5001:5001 getgit
   ```

3. **Run with environment variables (optional):**
   ```bash
   docker run -p 5001:5001 \
     -e GEMINI_API_KEY="your_api_key_here" \
     getgit
   ```
   Note: GEMINI_API_KEY is only needed if the local model fails to load

4. **Open your browser:**
   Navigate to `http://localhost:5001`

### Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python server.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5001`

## Environment Variables

- `GEMINI_API_KEY` - Optional. Used as fallback if local LLM fails (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- `FLASK_ENV` - Set to `development` to enable debug mode
- `PORT` - Server port (default: 5001)

## LLM Strategy

GetGit uses a dual-LLM strategy:

1. **Primary**: Local Hugging Face model (`Qwen/Qwen2.5-Coder-7B`)
   - Downloads automatically on first run
   - Cached in `./models` directory
   - Optimized for code understanding
   - Fully offline after initial download
   - No API costs

2. **Fallback**: Google Gemini (`gemini-2.5-flash`)
   - Used only if local model fails
   - Requires `GEMINI_API_KEY` environment variable
   - Provides reliable cloud-based alternative

## Repository Management

GetGit now intelligently manages repository state:

- **URL Persistence**: Current repository URL stored in `data/source_repo.txt`
- **Change Detection**: Automatically detects when a different repository is requested
- **Smart Cleanup**: Removes old repository data and re-indexes when URL changes
- **Efficient Reuse**: Reuses existing data when the same repository is accessed

This ensures:
- No stale embeddings
- No cross-repository contamination
- Deterministic repository state
- Efficient resource usage

## Requirements

- Python 3.9 or higher
- GitPython
- Flask
- numpy
- scikit-learn
- torch (for local LLM)
- transformers (for local LLM)
- accelerate (for optimized model loading)
- sentence-transformers (optional, for advanced embeddings)
- google-generativeai (optional, for Gemini fallback)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Documentation

For detailed technical documentation, architecture information, and API reference, see [documentation.md](documentation.md).
