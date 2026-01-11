"""
LLM connector module for RAG-based response generation.

This module provides integration with Large Language Models (LLMs) to generate
natural language responses based on retrieved repository context. It acts as
the generation component of the RAG pipeline, taking retrieved chunks and
user queries to produce synthesized answers.

The module is designed to be provider-agnostic, currently supporting Google's
Gemini models through the generativeai library, but can be extended to support
other LLM providers.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

# Try to import google.generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


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
    
    This function interfaces with Google's Gemini model to generate responses.
    It handles API configuration, error handling, and returns the generated text.
    
    Args:
        prompt: The formatted prompt to send to the LLM
        model_name: Name of the Gemini model to use (default: gemini-2.5-flash)
        api_key: Optional API key. If not provided, loads from GEMINI_API_KEY env var
    
    Returns:
        The LLM's generated response as plain text
    
    Raises:
        ImportError: If google-generativeai is not installed
        ValueError: If API key is not provided or found in environment
        Exception: If the API call fails
    
    Example:
        >>> response = query_llm("What is this repository about?")
    """
    if not GENAI_AVAILABLE:
        raise ImportError(
            "google-generativeai is not installed. "
            "Install it with: pip install google-generativeai"
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
        return response.text
    except Exception as e:
        raise Exception(f"Failed to generate response from LLM: {str(e)}")


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
