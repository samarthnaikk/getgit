#!/usr/bin/env python3
"""
Example script demonstrating how to use core.py for repository analysis.

This script shows various ways to use the GetGit core module for
analyzing GitHub repositories and answering questions about them.
"""

from core import main, initialize_repository, setup_rag, answer_query
from rag import RAGConfig
import sys


def example_basic_usage():
    """Example 1: Basic usage with default settings."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    result = main(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        query='What is this project about?',
        use_llm=False,  # Set to True if you have GEMINI_API_KEY set
        log_level='INFO'
    )
    
    print("\nResult:")
    print(f"- Retrieved {len(result['retrieved_chunks'])} relevant chunks")
    print(f"- Top match: {result['retrieved_chunks'][0]['file_path']}")


def example_custom_config():
    """Example 2: Using custom RAG configuration."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Configuration")
    print("="*70)
    
    # Use configuration optimized for documentation
    config = RAGConfig.for_documentation()
    
    result = main(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        query='How do I install this project?',
        use_llm=False,
        config=config,
        log_level='WARNING'  # Less verbose
    )
    
    print("\nDocumentation-focused search results:")
    for chunk in result['retrieved_chunks'][:3]:
        print(f"- {chunk['file_path']} (score: {chunk['score']:.3f})")


def example_step_by_step():
    """Example 3: Using individual functions for more control."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Step-by-Step Control")
    print("="*70)
    
    # Step 1: Initialize repository
    print("\n1. Initializing repository...")
    repo_path = initialize_repository(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        local_path='source_repo'
    )
    print(f"   Repository ready at: {repo_path}")
    
    # Step 2: Setup RAG
    print("\n2. Setting up RAG pipeline...")
    retriever = setup_rag(repo_path)
    print(f"   Indexed {len(retriever)} chunks")
    
    # Step 3: Process multiple queries
    print("\n3. Processing queries...")
    queries = [
        "What dependencies are required?",
        "How do I use the RAG system?",
        "What is the main entry point?"
    ]
    
    for query in queries:
        result = answer_query(
            query=query,
            retriever=retriever,
            use_llm=False,
            top_k=2
        )
        print(f"\n   Q: {query}")
        print(f"   A: Found {len(result['retrieved_chunks'])} relevant chunks")


def example_with_llm():
    """Example 4: Using LLM for natural language responses."""
    print("\n" + "="*70)
    print("EXAMPLE 4: With LLM Response Generation")
    print("="*70)
    
    # Check if API key is available
    import os
    if not os.getenv('GEMINI_API_KEY'):
        print("\nℹ GEMINI_API_KEY not set. Skipping LLM example.")
        print("  To enable LLM responses:")
        print("  1. Get API key from https://makersuite.google.com/app/apikey")
        print("  2. Set: export GEMINI_API_KEY=your_api_key_here")
        return
    
    result = main(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        query='Explain how the RAG pipeline works in this project',
        use_llm=True,
        top_k=5,
        log_level='WARNING'
    )
    
    print("\nLLM-Generated Response:")
    print("-" * 70)
    if result['response']:
        print(result['response'])
    else:
        print(f"Error: {result['error']}")


def main_menu():
    """Interactive menu for running examples."""
    print("\n" + "="*70)
    print("GetGit Core Module - Usage Examples")
    print("="*70)
    print("\nAvailable Examples:")
    print("1. Basic Usage")
    print("2. Custom Configuration")
    print("3. Step-by-Step Control")
    print("4. With LLM Response Generation")
    print("5. Run All Examples")
    print("0. Exit")
    
    choice = input("\nSelect an example (0-5): ").strip()
    
    examples = {
        '1': example_basic_usage,
        '2': example_custom_config,
        '3': example_step_by_step,
        '4': example_with_llm,
    }
    
    if choice == '5':
        for func in examples.values():
            func()
    elif choice in examples:
        examples[choice]()
    elif choice == '0':
        print("\nGoodbye!")
        return
    else:
        print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Run all examples non-interactively
        example_basic_usage()
        example_custom_config()
        example_step_by_step()
        example_with_llm()
    else:
        # Interactive mode
        main_menu()
