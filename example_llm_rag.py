"""
Example usage of the RAG system with LLM-based response generation.

This script demonstrates the complete RAG pipeline:
1. Clone a repository
2. Chunk the repository content
3. Create embeddings and index chunks
4. Perform natural language queries
5. Generate LLM-based responses using retrieved context

This showcases the integration of retrieval and generation for true
conversational question-answering over repositories.
"""

import os
import sys
from rag import (
    RepositoryChunker,
    SimpleEmbedding,
    Retriever,
    RAGConfig,
    generate_response
)
from clone_repo import clone_repo


def main():
    """Main example demonstrating RAG system with LLM integration."""
    
    # Step 1: Clone a repository (or use existing)
    repo_url = "https://github.com/samarthnaikk/getgit.git"
    repo_path = "source_repo"
    
    print("=" * 70)
    print("GetGit RAG + LLM System - Example Usage")
    print("=" * 70)
    
    if not os.path.exists(repo_path):
        print(f"\n1. Cloning repository from {repo_url}...")
        clone_repo(repo_url, repo_path)
        print(f"   > Repository cloned to {repo_path}")
    else:
        print(f"\n1. Using existing repository at {repo_path}")
    
    # Step 2: Initialize RAG configuration
    print("\n2. Initializing RAG configuration...")
    config = RAGConfig.default()
    print(f"   > Configuration loaded")
    
    # Step 3: Chunk the repository
    print("\n3. Chunking repository content...")
    chunker = RepositoryChunker(repo_path, repository_name="getgit")
    chunks = chunker.chunk_repository(config.chunking.file_patterns)
    print(f"   > Created {len(chunks)} chunks")
    
    # Step 4: Initialize embedding model
    print("\n4. Initializing embedding model...")
    if config.embedding.model_type == 'sentence-transformer':
        try:
            from rag import SentenceTransformerEmbedding
            embedding_model = SentenceTransformerEmbedding(config.embedding.model_name)
            print(f"   > Using SentenceTransformer: {config.embedding.model_name}")
        except ImportError:
            print("   ! sentence-transformers not available, falling back to SimpleEmbedding")
            embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
    else:
        embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
        print(f"   > Using SimpleEmbedding (TF-IDF based)")
    
    # Step 5: Create retriever and index chunks
    print("\n5. Creating retriever and indexing chunks...")
    retriever = Retriever(embedding_model)
    retriever.index_chunks(chunks, batch_size=config.embedding.batch_size)
    print(f"   > Indexed {len(retriever)} chunks")
    
    # Step 6: Check for API key
    print("\n6. Checking LLM configuration...")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("   ! GEMINI_API_KEY not found in environment")
        print("   ! To use LLM generation, set your API key:")
        print("     - Create a .env file with: GEMINI_API_KEY=your_api_key_here")
        print("     - Or export GEMINI_API_KEY=your_api_key_here")
        print("\n   Continuing with retrieval-only demonstration...")
        use_llm = False
    else:
        print("   > GEMINI_API_KEY found")
        use_llm = True
    
    # Step 7: Perform sample queries
    print("\n7. Performing sample queries with LLM generation...")
    print("=" * 70)
    
    queries = [
        "What is the main functionality of this project?",
        "How do I use the RAG system?",
        "What are the key features of GetGit?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'=' * 70}")
        print(f"Query {i}: '{query}'")
        print("=" * 70)
        
        # Retrieve relevant chunks
        print("\nRetrieving relevant context...")
        results = retriever.retrieve(query, top_k=5)
        
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for result in results[:3]:  # Show top 3
                print(f"  [{result.rank}] {result.chunk.file_path} (score: {result.score:.4f})")
            
            # Extract context for LLM
            context_chunks = [result.chunk.content for result in results]
            
            # Generate LLM response if API key is available
            if use_llm:
                print("\nGenerating LLM response...")
                try:
                    response = generate_response(query, context_chunks)
                    print("\n" + "-" * 70)
                    print("LLM Response:")
                    print("-" * 70)
                    print(response)
                    print("-" * 70)
                except Exception as e:
                    print(f"\n! Error generating LLM response: {str(e)}")
                    print("! Showing retrieved context instead:")
                    print("\nTop Retrieved Content:")
                    for result in results[:2]:
                        print(f"\n[{result.rank}] {result.chunk.file_path}")
                        content_preview = result.chunk.content[:300].replace('\n', ' ')
                        if len(result.chunk.content) > 300:
                            content_preview += "..."
                        print(content_preview)
            else:
                print("\nShowing retrieved context (LLM not configured):")
                for result in results[:2]:
                    print(f"\n[{result.rank}] {result.chunk.file_path}")
                    content_preview = result.chunk.content[:300].replace('\n', ' ')
                    if len(result.chunk.content) > 300:
                        content_preview += "..."
                    print(content_preview)
        else:
            print("  No relevant chunks found.")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
