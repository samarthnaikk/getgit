"""
Core orchestration module for GetGit RAG + LLM Pipeline.

This module serves as the unified entry point for GetGit, coordinating
repository cloning, RAG-based analysis, and LLM-powered question answering.
It provides a simple API for end-to-end repository intelligence gathering.
"""

import os
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from clone_repo import clone_repo
from rag import (
    RepositoryChunker,
    SimpleEmbedding,
    SentenceTransformerEmbedding,
    Retriever,
    RAGConfig,
    generate_response,
)
from checkpoints import (
    load_checkpoints,
    evaluate_checkpoint,
    run_checkpoints,
    format_results_summary,
    CheckpointResult
)


# Configure logging
def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the core module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('getgit.core')
    logger.setLevel(log_level)  # Explicitly set logger level
    return logger


# Initialize module logger
logger = setup_logging()


def initialize_repository(repo_url: str, local_path: str = "source_repo") -> str:
    """
    Clone or load the repository and prepare it for analysis.
    
    Args:
        repo_url: GitHub repository URL to clone
        local_path: Local path where repository will be stored
    
    Returns:
        Path to the cloned/loaded repository
    
    Raises:
        Exception: If repository cloning or loading fails
    """
    logger.info(f"Initializing repository from {repo_url}")
    
    try:
        if os.path.exists(local_path):
            logger.info(f"Repository already exists at {local_path}, using existing copy")
            logger.debug(f"Skipping clone for existing repository at {local_path}")
        else:
            logger.info(f"Cloning repository to {local_path}")
            clone_repo(repo_url, local_path)
            logger.info(f"Repository successfully cloned to {local_path}")
        
        # Verify repository exists and is accessible
        if not os.path.isdir(local_path):
            raise ValueError(f"Repository path {local_path} is not a valid directory")
        
        logger.debug(f"Repository initialized at {local_path}")
        return local_path
    
    except Exception as e:
        logger.error(f"Failed to initialize repository: {str(e)}")
        raise


def setup_rag(
    repo_path: str,
    repository_name: Optional[str] = None,
    config: Optional[RAGConfig] = None,
    use_sentence_transformer: bool = False
) -> Retriever:
    """
    Initialize chunker, embeddings, and retriever for RAG pipeline.
    
    Args:
        repo_path: Path to the repository to analyze
        repository_name: Optional name for the repository
        config: Optional RAG configuration (uses default if not provided)
        use_sentence_transformer: Whether to use SentenceTransformer embeddings
    
    Returns:
        Configured Retriever instance with indexed repository chunks
    
    Raises:
        Exception: If RAG initialization or indexing fails
    """
    logger.info(f"Setting up RAG pipeline for repository at {repo_path}")
    
    try:
        # Use default config if not provided
        if config is None:
            config = RAGConfig.default()
            logger.debug("Using default RAG configuration")
        
        # Determine repository name
        if repository_name is None:
            repository_name = os.path.basename(repo_path)
        logger.debug(f"Repository name: {repository_name}")
        
        # Step 1: Chunk the repository
        logger.info("Chunking repository content...")
        chunker = RepositoryChunker(repo_path, repository_name=repository_name)
        chunks = chunker.chunk_repository(config.chunking.file_patterns)
        logger.info(f"Created {len(chunks)} chunks from repository")
        
        if not chunks:
            logger.warning("No chunks created - repository may be empty or contain no supported file types")
            raise ValueError(
                "No chunks created from repository. Ensure the repository contains "
                f"files matching patterns: {config.chunking.file_patterns}"
            )
        
        # Step 2: Initialize embedding model
        logger.info("Initializing embedding model...")
        if use_sentence_transformer:
            try:
                embedding_model = SentenceTransformerEmbedding(config.embedding.model_name)
                logger.info(f"Using SentenceTransformer model: {config.embedding.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, falling back to SimpleEmbedding")
                embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
        else:
            embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
            logger.info("Using SimpleEmbedding (TF-IDF based)")
        
        # Step 3: Create retriever and index chunks
        logger.info("Creating retriever and indexing chunks...")
        retriever = Retriever(embedding_model)
        retriever.index_chunks(chunks, batch_size=config.embedding.batch_size)
        logger.info(f"Successfully indexed {len(retriever)} chunks")
        
        logger.debug("RAG pipeline setup complete")
        return retriever
    
    except Exception as e:
        logger.error(f"Failed to setup RAG pipeline: {str(e)}")
        raise


def answer_query(
    query: str,
    retriever: Retriever,
    top_k: int = 5,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict[str, Any]:
    """
    Retrieve relevant context and generate an LLM response for the query.
    
    Args:
        query: Natural language question about the repository
        retriever: Configured Retriever instance
        top_k: Number of relevant chunks to retrieve
        use_llm: Whether to generate LLM response (requires API key)
        api_key: Optional API key for LLM (reads from env if not provided)
        model_name: Name of the LLM model to use
    
    Returns:
        Dictionary containing:
            - query: The original query
            - retrieved_chunks: List of retrieved chunk information
            - context: Combined context from retrieved chunks
            - response: Generated LLM response (if use_llm=True)
            - error: Error message if LLM generation fails
    
    Raises:
        Exception: If query processing fails
    """
    logger.info(f"Processing query: '{query}'")
    
    try:
        # Step 1: Retrieve relevant chunks
        logger.info(f"Retrieving top {top_k} relevant chunks...")
        results = retriever.retrieve(query, top_k=top_k)
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        if not results:
            logger.warning("No relevant chunks found for query")
            return {
                'query': query,
                'retrieved_chunks': [],
                'context': '',
                'response': 'No relevant information found in the repository for this query.',
                'error': None
            }
        
        # Log retrieved chunks
        for result in results:
            logger.debug(
                f"Chunk {result.rank}: {result.chunk.file_path} "
                f"(score: {result.score:.4f}, type: {result.chunk.chunk_type.value})"
            )
        
        # Step 2: Extract context
        context_chunks = [result.chunk.content for result in results]
        retrieved_info = [
            {
                'rank': result.rank,
                'file_path': result.chunk.file_path,
                'chunk_type': result.chunk.chunk_type.value,
                'score': result.score,
                'start_line': result.chunk.start_line,
                'end_line': result.chunk.end_line,
                'metadata': result.chunk.metadata
            }
            for result in results
        ]
        
        # Step 3: Generate LLM response if requested
        response_text = None
        error = None
        
        if use_llm:
            logger.info("Generating LLM response...")
            try:
                response_text = generate_response(
                    query,
                    context_chunks,
                    model_name=model_name,
                    api_key=api_key
                )
                logger.info("LLM response generated successfully")
                logger.debug(f"Response length: {len(response_text)} characters")
            except Exception as e:
                error = str(e)
                logger.error(f"Failed to generate LLM response: {error}")
                response_text = None
        else:
            logger.debug("LLM response generation skipped (use_llm=False)")
        
        return {
            'query': query,
            'retrieved_chunks': retrieved_info,
            'context': '\n\n---\n\n'.join(context_chunks),
            'response': response_text,
            'error': error
        }
    
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}")
        raise


def validate_checkpoints(
    repo_url: str,
    checkpoints_file: str = "checkpoints.txt",
    local_path: str = "source_repo",
    use_llm: bool = True,
    log_level: str = "INFO",
    config: Optional[RAGConfig] = None,
    stop_on_failure: bool = False
) -> Dict[str, Any]:
    """
    Validate repository against checkpoints defined in a text file.
    
    This function orchestrates the checkpoint validation pipeline:
    1. Repository cloning/loading
    2. RAG initialization and indexing
    3. Loading checkpoints from file
    4. Sequential checkpoint evaluation
    5. Results aggregation and reporting
    
    Args:
        repo_url: GitHub repository URL
        checkpoints_file: Path to checkpoints text file
        local_path: Local path for repository storage
        use_llm: Whether to use LLM for checkpoint evaluation
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        config: Optional RAG configuration
        stop_on_failure: Stop processing on first checkpoint failure
    
    Returns:
        Dictionary containing:
            - checkpoints: List of checkpoint strings
            - results: List of CheckpointResult objects
            - summary: Formatted summary string
            - passed_count: Number of passed checkpoints
            - total_count: Total number of checkpoints
            - pass_rate: Percentage of passed checkpoints
    
    Raises:
        FileNotFoundError: If checkpoints file doesn't exist
        Exception: If any step of the pipeline fails
    
    Example:
        >>> result = validate_checkpoints(
        ...     repo_url="https://github.com/user/repo.git",
        ...     checkpoints_file="checkpoints.txt",
        ...     use_llm=True
        ... )
        >>> print(result['summary'])
    """
    # Setup logging
    global logger
    logger = setup_logging(log_level)
    
    logger.info("="*70)
    logger.info("GetGit Checkpoint Validation Pipeline Starting")
    logger.info("="*70)
    logger.info(f"Repository: {repo_url}")
    logger.info(f"Checkpoints File: {checkpoints_file}")
    logger.info(f"LLM Enabled: {use_llm}")
    logger.info("="*70)
    
    try:
        # Step 1: Initialize repository
        logger.info("\n[1/4] Initializing repository...")
        repo_path = initialize_repository(repo_url, local_path)
        logger.info(f"✓ Repository ready at {repo_path}")
        
        # Step 2: Setup RAG pipeline
        logger.info("\n[2/4] Setting up RAG pipeline...")
        retriever = setup_rag(repo_path, config=config)
        logger.info(f"✓ RAG pipeline ready with {len(retriever)} indexed chunks")
        
        # Step 3: Load checkpoints
        logger.info("\n[3/4] Loading checkpoints...")
        checkpoints = load_checkpoints(checkpoints_file)
        logger.info(f"✓ Loaded {len(checkpoints)} checkpoints")
        
        # Step 4: Run checkpoints
        logger.info("\n[4/4] Running checkpoint validation...")
        results = run_checkpoints(
            checkpoints=checkpoints,
            repo_path=repo_path,
            retriever=retriever,
            use_llm=use_llm,
            stop_on_failure=stop_on_failure
        )
        logger.info("✓ Checkpoint validation completed")
        
        # Generate summary
        summary = format_results_summary(results)
        
        # Calculate statistics
        passed_count = sum(1 for r in results if r.passed)
        total_count = len(results)
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0
        
        logger.info("\n" + "="*70)
        logger.info("GetGit Checkpoint Validation Pipeline Completed")
        logger.info(f"Results: {passed_count}/{total_count} passed ({pass_rate:.1f}%)")
        logger.info("="*70)
        
        return {
            'checkpoints': checkpoints,
            'results': results,
            'summary': summary,
            'passed_count': passed_count,
            'total_count': total_count,
            'pass_rate': pass_rate
        }
    
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("GetGit Checkpoint Validation Pipeline Failed")
        logger.error(f"Error: {str(e)}")
        logger.error("="*70)
        raise


def main(
    repo_url: str,
    query: str,
    local_path: str = "source_repo",
    use_llm: bool = True,
    top_k: int = 5,
    log_level: str = "INFO",
    config: Optional[RAGConfig] = None
) -> Dict[str, Any]:
    """
    Orchestrates the full GetGit pipeline from repository input to answer generation.
    
    This is the main entry point that coordinates:
    1. Repository cloning/loading
    2. RAG initialization and indexing
    3. Query processing and context retrieval
    4. LLM response generation
    
    Args:
        repo_url: GitHub repository URL
        query: Natural language question about the repository
        local_path: Local path for repository storage
        use_llm: Whether to generate LLM responses
        top_k: Number of relevant chunks to retrieve
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        config: Optional RAG configuration
    
    Returns:
        Dictionary containing query results and response
    
    Raises:
        Exception: If any step of the pipeline fails
    
    Example:
        >>> result = main(
        ...     repo_url="https://github.com/user/repo.git",
        ...     query="How do I install this project?",
        ...     use_llm=True
        ... )
        >>> print(result['response'])
    """
    # Setup logging
    global logger
    logger = setup_logging(log_level)
    
    logger.info("="*70)
    logger.info("GetGit Core Pipeline Starting")
    logger.info("="*70)
    logger.info(f"Repository: {repo_url}")
    logger.info(f"Query: {query}")
    logger.info(f"LLM Enabled: {use_llm}")
    logger.info("="*70)
    
    try:
        # Step 1: Initialize repository
        logger.info("\n[1/3] Initializing repository...")
        repo_path = initialize_repository(repo_url, local_path)
        logger.info(f"✓ Repository ready at {repo_path}")
        
        # Step 2: Setup RAG pipeline
        logger.info("\n[2/3] Setting up RAG pipeline...")
        retriever = setup_rag(repo_path, config=config)
        logger.info(f"✓ RAG pipeline ready with {len(retriever)} indexed chunks")
        
        # Step 3: Process query
        logger.info("\n[3/3] Processing query...")
        result = answer_query(
            query=query,
            retriever=retriever,
            top_k=top_k,
            use_llm=use_llm
        )
        logger.info("✓ Query processed successfully")
        
        logger.info("\n" + "="*70)
        logger.info("GetGit Core Pipeline Completed Successfully")
        logger.info("="*70)
        
        return result
    
    except Exception as e:
        logger.error("\n" + "="*70)
        logger.error("GetGit Core Pipeline Failed")
        logger.error(f"Error: {str(e)}")
        logger.error("="*70)
        raise


if __name__ == "__main__":
    """
    Example usage of the core module.
    
    This demonstrates a simple interactive session with GetGit.
    For CLI integration, consider using argparse or similar.
    """
    import sys
    
    # Example: Simple command-line usage
    if len(sys.argv) > 1:
        # If arguments provided, use them
        repo_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/samarthnaikk/getgit.git"
        query = sys.argv[2] if len(sys.argv) > 2 else "What is this project about?"
    else:
        # Default example
        repo_url = "https://github.com/samarthnaikk/getgit.git"
        query = "What is this project about?"
    
    print("\nGetGit - Repository Intelligence System")
    print("="*70)
    print(f"Repository: {repo_url}")
    print(f"Query: {query}")
    print("="*70 + "\n")
    
    try:
        # Run the pipeline
        result = main(
            repo_url=repo_url,
            query=query,
            use_llm=True,
            log_level="INFO"
        )
        
        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        print(f"\nQuery: {result['query']}")
        print(f"\nRetrieved {len(result['retrieved_chunks'])} relevant chunks:")
        for chunk_info in result['retrieved_chunks'][:3]:  # Show top 3
            print(f"  - {chunk_info['file_path']} (score: {chunk_info['score']:.4f})")
        
        if result['response']:
            print("\n" + "-"*70)
            print("ANSWER:")
            print("-"*70)
            print(result['response'])
        elif result['error']:
            print("\n" + "-"*70)
            print("ERROR:")
            print("-"*70)
            print(f"Failed to generate LLM response: {result['error']}")
            print("\nShowing retrieved context instead:")
            print("-"*70)
            # Show snippet of context
            context_preview = result['context'][:500]
            if len(result['context']) > 500:
                context_preview += "..."
            print(context_preview)
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
