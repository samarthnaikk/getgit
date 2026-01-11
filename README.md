# GetGit

GetGit is a repository intelligence system that helps you analyze and understand GitHub repositories using natural language. It combines Retrieval-Augmented Generation (RAG) with Large Language Models to provide intelligent answers about any codebase.

## What GetGit Does

- **Repository Analysis**: Automatically indexes and analyzes GitHub repositories
- **Natural Language Queries**: Ask questions about code in plain English
- **Checkpoint Validation**: Programmatically validate repository requirements
- **Intelligent Search**: Semantic search across code, documentation, and configuration files

## Features

- **RAG-Based Analysis**: Semantic chunking and retrieval of repository content
- **LLM Integration**: Natural language response generation using Google Gemini
- **Checkpoint System**: Automated validation of repository requirements
- **Web Interface**: User-friendly interface for repository exploration

## How to Run

### Using Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t getgit .
   ```

2. **Run the container:**
   ```bash
   docker run -p 5000:5000 getgit
   ```

3. **Run with environment variables:**
   ```bash
   docker run -p 5000:5000 \
     -e GEMINI_API_KEY="your_api_key_here" \
     -e FLASK_SECRET_KEY="your_secret_key_here" \
     getgit
   ```

4. **Open your browser:**
   Navigate to `http://localhost:5000`

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
   Navigate to `http://localhost:5000`

## Environment Variables

- `GEMINI_API_KEY` - Required for LLM-powered responses (get from [Google AI Studio](https://makersuite.google.com/app/apikey))
- `FLASK_SECRET_KEY` - Secret key for Flask sessions (set in production)
- `FLASK_ENV` - Set to `development` to enable debug mode

## Requirements

- Python 3.6 or higher
- GitPython
- Flask
- numpy
- scikit-learn
- sentence-transformers (optional, for advanced embeddings)
- google-generativeai (optional, for LLM responses)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Documentation

For detailed technical documentation, architecture information, and API reference, see [documentation.md](documentation.md).
