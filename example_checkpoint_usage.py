#!/usr/bin/env python3
"""
Example script demonstrating checkpoint validation system usage.

This script shows how to use the checkpoint validation feature to
programmatically validate repository requirements.
"""

from core import validate_checkpoints, initialize_repository, setup_rag
from checkpoints import load_checkpoints, run_checkpoints, format_results_summary
import sys
import os


def example_basic_checkpoint_validation():
    """Example 1: Basic checkpoint validation using validate_checkpoints()."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Checkpoint Validation")
    print("="*70)
    
    result = validate_checkpoints(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        checkpoints_file='checkpoints.txt',
        use_llm=False,  # Set to True if you have GEMINI_API_KEY set
        log_level='INFO'
    )
    
    print("\nValidation Results:")
    print(result['summary'])
    print(f"\nOverall: {result['passed_count']}/{result['total_count']} checkpoints passed")


def example_custom_checkpoints():
    """Example 2: Using custom checkpoint file."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Checkpoints")
    print("="*70)
    
    # Create a temporary checkpoint file
    custom_checkpoints = """
# Custom validation checkpoints
1. Check if documentation.md exists
2. Check if the project uses Python
3. Check if there is a requirements.txt file
""".strip()
    
    checkpoint_file = '/tmp/custom_checkpoints.txt'
    with open(checkpoint_file, 'w') as f:
        f.write(custom_checkpoints)
    
    result = validate_checkpoints(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        checkpoints_file=checkpoint_file,
        use_llm=False,
        log_level='WARNING'  # Less verbose
    )
    
    print("\nCustom Validation Results:")
    for i, checkpoint_result in enumerate(result['results'], 1):
        print(f"{i}. {checkpoint_result.format_output()}")
    
    # Clean up
    os.remove(checkpoint_file)


def example_step_by_step_validation():
    """Example 3: Manual step-by-step checkpoint validation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Step-by-Step Validation")
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
    
    # Step 3: Load and run checkpoints
    print("\n3. Loading and running checkpoints...")
    checkpoints = load_checkpoints('checkpoints.txt')
    print(f"   Loaded {len(checkpoints)} checkpoints")
    
    results = run_checkpoints(
        checkpoints=checkpoints,
        repo_path=repo_path,
        retriever=retriever,
        use_llm=False,
        stop_on_failure=False
    )
    
    # Step 4: Display results
    print("\n4. Validation Results:")
    summary = format_results_summary(results)
    print(summary)


def example_with_llm():
    """Example 4: Checkpoint validation with LLM interpretation."""
    print("\n" + "="*70)
    print("EXAMPLE 4: With LLM Interpretation")
    print("="*70)
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("\nℹ GEMINI_API_KEY not set. Skipping LLM example.")
        print("  To enable LLM-based validation:")
        print("  1. Get API key from https://makersuite.google.com/app/apikey")
        print("  2. Set: export GEMINI_API_KEY=your_api_key_here")
        return
    
    result = validate_checkpoints(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        checkpoints_file='checkpoints.txt',
        use_llm=True,
        log_level='WARNING'
    )
    
    print("\nLLM-Enhanced Validation Results:")
    print(result['summary'])
    
    # Show detailed results with LLM explanations
    print("\nDetailed Results:")
    for i, checkpoint_result in enumerate(result['results'], 1):
        print(f"\n{i}. {checkpoint_result.format_output()}")


def example_stop_on_failure():
    """Example 5: Fast-fail validation mode."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Stop on First Failure")
    print("="*70)
    
    result = validate_checkpoints(
        repo_url='https://github.com/samarthnaikk/getgit.git',
        checkpoints_file='checkpoints.txt',
        use_llm=False,
        log_level='INFO',
        stop_on_failure=True  # Stop on first failure
    )
    
    print("\nFast-fail Results:")
    print(f"Evaluated: {result['total_count']} checkpoints")
    print(f"Passed: {result['passed_count']}")
    
    if result['passed_count'] < len(load_checkpoints('checkpoints.txt')):
        print("⚠ Stopped early due to failure")


def main_menu():
    """Interactive menu for running examples."""
    print("\n" + "="*70)
    print("GetGit Checkpoint Validation - Usage Examples")
    print("="*70)
    print("\nAvailable Examples:")
    print("1. Basic Checkpoint Validation")
    print("2. Custom Checkpoints")
    print("3. Step-by-Step Validation")
    print("4. With LLM Interpretation")
    print("5. Stop on First Failure")
    print("6. Run All Examples")
    print("0. Exit")
    
    choice = input("\nSelect an example (0-6): ").strip()
    
    examples = {
        '1': example_basic_checkpoint_validation,
        '2': example_custom_checkpoints,
        '3': example_step_by_step_validation,
        '4': example_with_llm,
        '5': example_stop_on_failure,
    }
    
    if choice == '6':
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
        example_basic_checkpoint_validation()
        example_custom_checkpoints()
        example_step_by_step_validation()
        example_with_llm()
        example_stop_on_failure()
    else:
        # Interactive mode
        main_menu()
