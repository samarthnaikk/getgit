"""
Example usage of the RAG system for repository analysis.

This script demonstrates how to:
1. Clone a repository
2. Chunk the repository content
3. Create embeddings and index chunks
4. Perform natural language queries
"""

import os
import sys
from rag import (
    RepositoryChunker, 
    SimpleEmbedding, 
    Retriever, 
    RAGConfig
)
from clone_repo import clone_repo


def main():
    """Main example demonstrating RAG system usage."""
    
    # Step 1: Clone a repository (or use existing)
    repo_url = "https://github.com/samarthnaikk/getgit.git"
    repo_path = "source_repo"
    
    print("=" * 70)
    print("GetGit RAG System - Example Usage")
    print("=" * 70)
    
    if not os.path.exists(repo_path):
        print(f"\n1. Cloning repository from {repo_url}...")
        clone_repo(repo_url, repo_path)
        print(f"   ✓ Repository cloned to {repo_path}")
    else:
        print(f"\n1. Using existing repository at {repo_path}")
    
    # Step 2: Initialize RAG configuration
    print("\n2. Initializing RAG configuration...")
    config = RAGConfig.default()
    print(f"   ✓ Configuration loaded")
    print(f"   - File patterns: {config.chunking.file_patterns}")
    print(f"   - Embedding model: {config.embedding.model_type}")
    
    # Step 3: Chunk the repository
    print("\n3. Chunking repository content...")
    chunker = RepositoryChunker(repo_path, repository_name="getgit")
    chunks = chunker.chunk_repository(config.chunking.file_patterns)
    print(f"   ✓ Created {len(chunks)} chunks")
    
    # Display sample chunks
    if chunks:
        print("\n   Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"   - Chunk {i+1}: {chunk.chunk_type.value} from {chunk.file_path}")
            print(f"     Lines {chunk.start_line}-{chunk.end_line}")
            if chunk.metadata:
                print(f"     Metadata: {chunk.metadata}")
    
    # Step 4: Initialize embedding model
    print("\n4. Initializing embedding model...")
    if config.embedding.model_type == 'sentence-transformer':
        try:
            from rag import SentenceTransformerEmbedding
            embedding_model = SentenceTransformerEmbedding(config.embedding.model_name)
            print(f"   ✓ Using SentenceTransformer: {config.embedding.model_name}")
        except ImportError:
            print("   ⚠ sentence-transformers not available, falling back to SimpleEmbedding")
            embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
    else:
        embedding_model = SimpleEmbedding(max_features=config.embedding.embedding_dim)
        print(f"   ✓ Using SimpleEmbedding (TF-IDF based)")
    
    # Step 5: Create retriever and index chunks
    print("\n5. Creating retriever and indexing chunks...")
    retriever = Retriever(embedding_model)
    retriever.index_chunks(chunks, batch_size=config.embedding.batch_size)
    print(f"   ✓ Indexed {len(retriever)} chunks")
    
    # Step 6: Perform sample queries
    print("\n6. Performing sample queries...")
    print("=" * 70)
    
    queries = [
        "How do I clone a repository?",
        "What is the main functionality of this project?",
        "repository cloning function"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        results = retriever.retrieve(query, top_k=3)
        
        if results:
            for result in results:
                print(f"\n  [{result.rank}] Score: {result.score:.4f}")
                print(f"      Type: {result.chunk.chunk_type.value}")
                print(f"      File: {result.chunk.file_path}")
                print(f"      Lines: {result.chunk.start_line}-{result.chunk.end_line}")
                if result.chunk.metadata:
                    print(f"      Metadata: {result.chunk.metadata}")
                
                # Show snippet of content
                content_preview = result.chunk.content[:200].replace('\n', ' ')
                if len(result.chunk.content) > 200:
                    content_preview += "..."
                print(f"      Preview: {content_preview}")
        else:
            print("  No results found.")
    
    # Step 7: Optional - Save retriever for later use
    print("\n" + "=" * 70)
    print("\n7. Saving retriever state...")
    cache_dir = config.retrieval.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    retriever_path = os.path.join(cache_dir, "retriever.pkl")
    retriever.save(retriever_path)
    print(f"   ✓ Retriever saved to {retriever_path}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
