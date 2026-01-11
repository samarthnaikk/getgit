"""
LLM connector module for RAG-based response generation.

This module provides integration with Large Language Models (LLMs) to generate
natural language responses based on retrieved repository context. It acts as
the generation component of the RAG pipeline, taking retrieved chunks and
user queries to produce synthesized answers.

The module supports:
1. Local Hugging Face models (primary): Qwen/Qwen2.5-Coder-7B
2. Google Gemini models (fallback): gemini-2.5-flash

The local model is prioritized for offline usage, privacy, and code understanding.
Gemini is used as an automatic fallback if local model loading or inference fails.
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger('getgit.llm_connector')

# Try to import transformers for local LLM
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, local LLM will not be available")

# Try to import google.generativeai for Gemini fallback
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-generativeai not available, Gemini fallback will not be available")


# Global cache for local model
_local_model = None
_local_tokenizer = None
_local_model_failed = False


def load_local_model(model_name: str = "Qwen/Qwen2.5-Coder-7B") -> tuple:
    """
    Load the local Hugging Face model.
    
    Args:
        model_name: Name of the model to load from Hugging Face
    
    Returns:
        Tuple of (tokenizer, model) if successful, (None, None) if failed
    """
    global _local_model, _local_tokenizer, _local_model_failed
    
    # Return cached model if available
    if _local_model is not None and _local_tokenizer is not None:
        logger.debug("Using cached local model")
        return _local_tokenizer, _local_model
    
    # Don't retry if previous attempt failed
    if _local_model_failed:
        logger.debug("Previous local model load failed, skipping")
        return None, None
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("transformers not available, cannot load local model")
        _local_model_failed = True
        return None, None
    
    try:
        logger.info(f"Loading local model: {model_name}")
        logger.info("This may take a few minutes on first run...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./models"
        )
        
        # Load model with automatic device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            cache_dir="./models"
        )
        
        # Move to CPU if CUDA is not available
        if not torch.cuda.is_available():
            model = model.to('cpu')
            logger.info("Running model on CPU (CUDA not available)")
        else:
            logger.info(f"Running model on GPU")
        
        # Cache the model
        _local_model = model
        _local_tokenizer = tokenizer
        
        logger.info(f"Successfully loaded local model: {model_name}")
        return tokenizer, model
    
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        _local_model_failed = True
        return None, None


def query_local_llm(prompt: str, model_name: str = "Qwen/Qwen2.5-Coder-7B",
                   max_new_tokens: int = 1024) -> Optional[str]:
    """
    Query the local Hugging Face model.
    
    Args:
        prompt: The formatted prompt to send to the LLM
        model_name: Name of the model to use
        max_new_tokens: Maximum number of tokens to generate
    
    Returns:
        Generated response text if successful, None if failed
    """
    try:
        tokenizer, model = load_local_model(model_name)
        
        if tokenizer is None or model is None:
            logger.warning("Local model not available")
            return None
        
        logger.info("Generating response with local model...")
        
        # Prepare the input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated text (remove the prompt)
        response = full_response[len(prompt):].strip()
        
        logger.info("Local model response generated successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error querying local model: {str(e)}")
        return None


def build_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Combines user query and retrieved context into a single prompt.
    
    This function constructs a well-formatted prompt that provides the LLM
    with relevant context from the repository and the user's question.
    
    Args:
        query: The user's natural language question
        context_chunks: List of retrieved text chunks from the repository
    
    Returns:
        A formatted prompt string ready to be sent to the LLM
    
    Example:
        >>> chunks = ["def clone_repo(url): ...", "# Repository cloning utility"]
        >>> prompt = build_prompt("How do I clone a repo?", chunks)
    """
    if not context_chunks:
        return f"""You are a helpful assistant that answers questions about a code repository.

User Question: {query}

Note: No relevant context was found in the repository. Please provide a general answer or indicate that you need more information."""
    
    # Combine context chunks into a single context block
    context = "\n\n---\n\n".join(context_chunks)
    
    # Build the full prompt
    prompt = f"""You are a helpful assistant that answers questions about a code repository based on the provided context.

Context from Repository:
{context}

---

User Question: {query}

Please provide a clear, concise answer based on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge this and provide what information you can."""
    
    return prompt


def query_llm(prompt: str, model_name: str = "gemini-2.5-flash", 
              api_key: Optional[str] = None) -> str:
    """
    Sends the prompt to an LLM and returns the generated response.
    
    This function first attempts to use the local Hugging Face model.
    If local model is unavailable or fails, it automatically falls back to Gemini.
    
    Args:
        prompt: The formatted prompt to send to the LLM
        model_name: Name of the Gemini model to use as fallback (default: gemini-2.5-flash)
        api_key: Optional API key for Gemini. If not provided, loads from GEMINI_API_KEY env var
    
    Returns:
        The LLM's generated response as plain text
    
    Raises:
        Exception: If both local model and Gemini fallback fail
    
    Example:
        >>> response = query_llm("What is this repository about?")
    """
    # First, try local model
    logger.info("Attempting to use local Hugging Face model...")
    local_response = query_local_llm(prompt)
    
    if local_response is not None:
        logger.info("Successfully used local model")
        return local_response
    
    # Fallback to Gemini
    logger.info("Local model unavailable, falling back to Gemini...")
    
    if not GENAI_AVAILABLE:
        raise ImportError(
            "Neither local model nor google-generativeai is available. "
            "Install transformers and torch for local model, or "
            "install google-generativeai for Gemini fallback."
        )
    
    # Load environment variables from .env file if present
    load_dotenv()
    
    # Get API key from parameter or environment
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Please provide it as a parameter "
            "or set it in your environment variables or .env file."
        )
    
    # Configure the generativeai library
    genai.configure(api_key=api_key)
    # Always use gemini-2.5-flash as the model name
    model_name = "gemini-2.5-flash"
    try:
        # Initialize the model
        model = genai.GenerativeModel(model_name)
        # Generate response
        response = model.generate_content(prompt)
        # Extract and return the text
        logger.info("Successfully used Gemini fallback")
        return response.text
    except Exception as e:
        raise Exception(f"Failed to generate response from LLM (both local and Gemini): {str(e)}")


def generate_response(query: str, context_chunks: List[str], 
                      model_name: str = "gemini-2.5-flash",
                      api_key: Optional[str] = None) -> str:
    """
    High-level function that builds the prompt, queries the LLM,
    and returns the final response.
    
    This is the main entry point for generating LLM-based responses in the
    RAG pipeline. It combines the prompt building and LLM querying steps
    into a single convenient function.
    
    Args:
        query: The user's natural language question
        context_chunks: List of retrieved text chunks from the repository
        model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        api_key: Optional API key. If not provided, loads from GEMINI_API_KEY env var
    
    Returns:
        The LLM's generated response as plain text
    
    Raises:
        ImportError: If google-generativeai is not installed
        ValueError: If API key is not provided or found in environment
        Exception: If the API call fails
    
    Example:
        >>> from rag import Retriever, SimpleEmbedding
        >>> retriever = Retriever(SimpleEmbedding())
        >>> # ... index chunks ...
        >>> results = retriever.retrieve("How do I clone a repository?")
        >>> context = [r.chunk.content for r in results]
        >>> response = generate_response("How do I clone a repository?", context)
        >>> print(response)
    """
    # Build the prompt from query and context
    prompt = build_prompt(query, context_chunks)
    # Always use gemini-2.5-flash as the model name
    return query_llm(prompt, model_name="gemini-2.5-flash", api_key=api_key)
