# GetGit - Local LLM Usage Guide

This guide explains the new local LLM features in GetGit and how to use them.

## Overview

GetGit now supports running a local coding-optimized LLM (Qwen/Qwen2.5-Coder-7B) directly on your machine, with automatic fallback to Google Gemini if needed.

## Key Features

### 1. Local LLM (Primary)
- **Model**: Qwen/Qwen2.5-Coder-7B from Hugging Face
- **First Run**: Automatically downloads (~14GB) and caches in `./models/`
- **Subsequent Runs**: Uses cached model (fully offline)
- **Optimized For**: Code understanding, generation, and analysis
- **No API Key Required**: Completely free and private

### 2. Gemini Fallback (Automatic)
- **Trigger**: Only if local model fails to load or generate
- **Model**: gemini-2.5-flash
- **Requires**: `GEMINI_API_KEY` environment variable
- **Use Case**: Backup for systems without sufficient resources

### 3. Repository Persistence
- **Tracking**: Current repository URL stored in `data/source_repo.txt`
- **Change Detection**: Automatically detects when a different repo is requested
- **Smart Cleanup**: Removes old data only when necessary
- **Efficiency**: Reuses existing data for the same repository

## Quick Start

### Using Docker (Recommended)

1. **Build the image:**
   ```bash
   docker build -t getgit .
   ```

2. **Run without Gemini (local model only):**
   ```bash
   docker run -p 5001:5001 getgit
   ```
   
   The local model will download on first run (~10-15 minutes depending on connection).

3. **Run with Gemini fallback (optional):**
   ```bash
   docker run -p 5001:5001 \
     -e GEMINI_API_KEY="your_api_key_here" \
     getgit
   ```

4. **Access the web UI:**
   ```
   http://localhost:5001
   ```

### Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   ```bash
   python server.py
   ```

3. **Access the web UI:**
   ```
   http://localhost:5001
   ```

## Model Download

On first run, the local model will be downloaded automatically:

```
INFO - Loading local model: Qwen/Qwen2.5-Coder-7B
INFO - This may take a few minutes on first run...
INFO - Successfully loaded local model
```

**Download Size**: ~14GB  
**Cache Location**: `./models/`  
**Reusable**: Yes, persists across restarts

## System Requirements

### Minimum (CPU Mode)
- **RAM**: 16GB
- **Storage**: 20GB free
- **CPU**: Multi-core processor

### Recommended (GPU Mode)
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **Storage**: 20GB free
- **CUDA**: 11.7 or higher

## LLM Selection Logic

The system automatically selects the best available LLM:

```
1. Attempt local Hugging Face model
   ├─ Success → Use local model
   └─ Failure → Try Gemini fallback
       ├─ API key available → Use Gemini
       └─ No API key → Error
```

**Note**: The fallback is automatic and transparent to the user.

## Repository Management

### How It Works

1. **First Repository**:
   ```
   POST /initialize {"repo_url": "https://github.com/user/repo1.git"}
   → Clones repo1
   → Stores URL in data/source_repo.txt
   → Indexes content
   ```

2. **Same Repository Again**:
   ```
   POST /initialize {"repo_url": "https://github.com/user/repo1.git"}
   → Detects same URL
   → Reuses existing clone and index
   → Fast startup
   ```

3. **Different Repository**:
   ```
   POST /initialize {"repo_url": "https://github.com/user/repo2.git"}
   → Detects URL change
   → Deletes source_repo/ directory
   → Deletes .rag_cache/ directory
   → Updates data/source_repo.txt
   → Clones repo2
   → Re-indexes from scratch
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | - | Fallback API key for Gemini |
| `PORT` | No | 5001 | Server port |
| `FLASK_ENV` | No | production | Flask environment |

## Troubleshooting

### Local Model Won't Load

**Symptom**: "Local model unavailable, falling back to Gemini..."

**Solutions**:
1. Check available RAM (need 16GB+)
2. Check available storage (need 20GB+)
3. Verify transformers/torch are installed
4. Check logs for specific error message

### Out of Memory

**Symptom**: Process killed or memory error during model load

**Solutions**:
1. Close other applications
2. Use smaller model (requires code changes)
3. Use Gemini fallback instead
4. Add more RAM or swap space

### Model Download Fails

**Symptom**: Connection errors during first run

**Solutions**:
1. Check internet connection
2. Check firewall settings
3. Retry (downloads resume automatically)
4. Use manual download and place in `./models/`

### Repository Not Updating

**Symptom**: Old repository content shown for new URL

**Solutions**:
1. Delete `data/source_repo.txt`
2. Delete `source_repo/` directory
3. Delete `.rag_cache/` directory
4. Restart application

## Performance Tips

1. **First Run**: Expect 10-15 minute model download
2. **Subsequent Runs**: Model loads in ~30-60 seconds
3. **GPU Usage**: Automatically detected and used if available
4. **CPU Usage**: Works but slower (~5-10x slower than GPU)
5. **Memory**: Keep 16GB+ free for optimal performance

## Security

- **Local Model**: No data sent externally
- **Gemini Fallback**: Only used if explicitly configured
- **API Keys**: Never logged or stored in code
- **Privacy**: Local mode is completely offline

## Limitations

1. **Model Size**: 7B parameters (large but manageable)
2. **Context Length**: 4096 tokens max
3. **GPU Memory**: Requires 8GB+ VRAM for best performance
4. **First Run**: Requires internet for model download

## Support

For issues or questions:
1. Check logs for error messages
2. Review troubleshooting section above
3. Open an issue on GitHub
4. Include system specs and error logs
