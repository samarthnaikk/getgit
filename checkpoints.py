"""
Checkpoint-based validation system for repository analysis.

This module provides functionality to validate repository requirements using
checkpoint definitions from a text file. Each checkpoint represents a requirement
that is automatically evaluated using repository analysis and RAG capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from rag import Retriever, generate_response


# Module logger
logger = logging.getLogger('getgit.checkpoints')


class CheckpointResult:
    """
    Result from evaluating a single checkpoint.
    
    Attributes:
        checkpoint: The original checkpoint text
        passed: Whether the checkpoint passed validation
        explanation: Detailed explanation of the result
        evidence: Supporting files or information
        score: Optional confidence score (0.0-1.0)
    """
    
    def __init__(
        self,
        checkpoint: str,
        passed: bool,
        explanation: str,
        evidence: Optional[List[str]] = None,
        score: Optional[float] = None
    ):
        self.checkpoint = checkpoint
        self.passed = passed
        self.explanation = explanation
        self.evidence = evidence or []
        self.score = score
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"CheckpointResult({status}, checkpoint='{self.checkpoint[:50]}...')"
    
    def format_output(self) -> str:
        """Format the result as human-readable text."""
        status = "[PASS]" if self.passed else "[FAIL]"
        output = f"{status} {self.checkpoint}\n"
        output += f"  {self.explanation}\n"
        if self.evidence:
            output += f"  Evidence: {', '.join(self.evidence)}\n"
        if self.score is not None:
            output += f"  Confidence: {self.score:.2f}\n"
        return output


def load_checkpoints(file_path: str) -> List[str]:
    """
    Load and parse checkpoint definitions from a text file.
    
    The file should contain one checkpoint per line, optionally numbered.
    Empty lines and lines starting with '#' are ignored.
    
    Args:
        file_path: Path to the checkpoints file
    
    Returns:
        List of checkpoint strings
    
    Raises:
        FileNotFoundError: If the checkpoints file doesn't exist
        ValueError: If the file is empty or contains no valid checkpoints
    
    Example:
        >>> checkpoints = load_checkpoints('checkpoints.txt')
        >>> print(checkpoints[0])
        Check if the repository has README.md
    """
    logger.info(f"Loading checkpoints from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Checkpoints file not found: {file_path}")
    
    checkpoints = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Remove numbering if present (e.g., "1. ", "1) ", "1 - ")
            checkpoint = re.sub(r'^\d+[\.\)\-\:]\s*', '', line)
            
            if checkpoint:
                checkpoints.append(checkpoint)
                logger.debug(f"Loaded checkpoint {len(checkpoints)}: {checkpoint[:50]}...")
    
    if not checkpoints:
        raise ValueError(f"No valid checkpoints found in {file_path}")
    
    logger.info(f"Loaded {len(checkpoints)} checkpoints")
    return checkpoints


def _check_file_exists(checkpoint: str, repo_path: str) -> Optional[CheckpointResult]:
    """
    Check if a checkpoint is asking about file existence and handle it deterministically.
    
    Args:
        checkpoint: The checkpoint text
        repo_path: Path to the repository
    
    Returns:
        CheckpointResult if it's a file existence check, None otherwise
    """
    # Pattern matching for file existence checks
    # Look for common filenames with extensions
    file_pattern = r'\b([\w\-]+\.[\w]+)\b'
    
    matches = re.findall(file_pattern, checkpoint)
    
    # Check if this is actually asking about file existence
    existence_keywords = ['check if', 'has', 'contains', 'includes', 'exists', 'present', 'available']
    is_existence_check = any(keyword in checkpoint.lower() for keyword in existence_keywords)
    
    if matches and is_existence_check:
        # Use the first filename found
        filename = matches[0]
        
        # Search for the file in the repository
        found_files = []
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.lower() == filename.lower():
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    found_files.append(rel_path)
        
        if found_files:
            return CheckpointResult(
                checkpoint=checkpoint,
                passed=True,
                explanation=f"File '{filename}' found in repository",
                evidence=found_files,
                score=1.0
            )
        else:
            return CheckpointResult(
                checkpoint=checkpoint,
                passed=False,
                explanation=f"File '{filename}' not found in repository",
                evidence=[],
                score=1.0
            )
    
    return None


def evaluate_checkpoint(
    checkpoint: str,
    repo_path: str,
    retriever: Retriever,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> CheckpointResult:
    """
    Evaluate a single checkpoint and return result details.
    
    The evaluation process:
    1. Try deterministic checks first (e.g., file existence)
    2. Use RAG retrieval to find relevant context
    3. Optionally use LLM to interpret complex requirements
    
    Args:
        checkpoint: The checkpoint requirement to evaluate
        repo_path: Path to the repository
        retriever: Configured Retriever instance for RAG
        use_llm: Whether to use LLM for evaluation
        api_key: Optional API key for LLM
        model_name: Name of the LLM model to use
    
    Returns:
        CheckpointResult with evaluation outcome
    
    Example:
        >>> result = evaluate_checkpoint(
        ...     "Check if README.md exists",
        ...     "/path/to/repo",
        ...     retriever
        ... )
        >>> print(result.format_output())
    """
    logger.info(f"Evaluating checkpoint: {checkpoint[:50]}...")
    
    # Step 1: Try deterministic checks
    file_check = _check_file_exists(checkpoint, repo_path)
    if file_check:
        logger.info(f"Checkpoint evaluated deterministically: {'PASS' if file_check.passed else 'FAIL'}")
        return file_check
    
    # Step 2: Use RAG retrieval
    logger.debug("Using RAG retrieval for checkpoint evaluation")
    try:
        results = retriever.retrieve(checkpoint, top_k=5)
        
        if not results:
            return CheckpointResult(
                checkpoint=checkpoint,
                passed=False,
                explanation="No relevant information found in repository",
                evidence=[],
                score=0.0
            )
        
        # Collect evidence
        evidence_files = [result.chunk.file_path for result in results[:3]]
        context_chunks = [result.chunk.content for result in results]
        
        # Step 3: Use LLM for interpretation if available
        if use_llm:
            try:
                # Create a specialized prompt for checkpoint evaluation
                eval_prompt = f"""Based on the following repository context, evaluate this requirement:

Requirement: {checkpoint}

Repository Context:
{chr(10).join(f"--- Chunk {i+1} ---{chr(10)}{chunk}" for i, chunk in enumerate(context_chunks[:3]))}

Provide a clear evaluation:
1. Does the repository satisfy this requirement? (Yes/No)
2. Explain your reasoning in 1-2 sentences
3. If applicable, mention specific files or components that demonstrate this

Format your response as:
RESULT: [Yes/No]
EXPLANATION: [Your explanation]
"""
                
                response = generate_response(
                    eval_prompt,
                    context_chunks,
                    model_name=model_name,
                    api_key=api_key
                )
                
                # Parse LLM response
                passed = "yes" in response.lower()[:100]  # Check beginning of response
                explanation_match = re.search(r'EXPLANATION:\s*(.+?)(?:\n\n|\Z)', response, re.DOTALL)
                
                if explanation_match:
                    explanation = explanation_match.group(1).strip()
                else:
                    explanation = response[:200] + "..." if len(response) > 200 else response
                
                # Calculate score based on retrieval scores
                avg_score = sum(r.score for r in results[:3]) / min(3, len(results))
                
                return CheckpointResult(
                    checkpoint=checkpoint,
                    passed=passed,
                    explanation=explanation,
                    evidence=evidence_files,
                    score=avg_score
                )
                
            except Exception as e:
                logger.warning(f"LLM evaluation failed: {e}, falling back to RAG-only")
        
        # Fallback: Use retrieval scores only
        # If top result has high score, consider it a pass
        top_score = results[0].score
        threshold = 0.5  # Configurable threshold
        
        passed = top_score >= threshold
        explanation = f"Found relevant content (score: {top_score:.2f}). "
        if passed:
            explanation += f"Repository likely satisfies this requirement based on {len(results)} relevant chunks."
        else:
            explanation += f"Insufficient evidence found. Relevance score below threshold ({threshold})."
        
        return CheckpointResult(
            checkpoint=checkpoint,
            passed=passed,
            explanation=explanation,
            evidence=evidence_files,
            score=top_score
        )
        
    except Exception as e:
        logger.error(f"Error evaluating checkpoint: {e}")
        return CheckpointResult(
            checkpoint=checkpoint,
            passed=False,
            explanation=f"Evaluation error: {str(e)}",
            evidence=[],
            score=0.0
        )


def run_checkpoints(
    checkpoints: List[str],
    repo_path: str,
    retriever: Retriever,
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    stop_on_failure: bool = False
) -> List[CheckpointResult]:
    """
    Run all checkpoints and return aggregated results.
    
    Evaluates each checkpoint sequentially and collects results.
    Optionally stops on first failure for fast-fail scenarios.
    
    Args:
        checkpoints: List of checkpoint requirements
        repo_path: Path to the repository
        retriever: Configured Retriever instance
        use_llm: Whether to use LLM for evaluation
        api_key: Optional API key for LLM
        model_name: Name of the LLM model to use
        stop_on_failure: Stop processing on first failure
    
    Returns:
        List of CheckpointResult objects
    
    Example:
        >>> checkpoints = load_checkpoints('checkpoints.txt')
        >>> results = run_checkpoints(checkpoints, repo_path, retriever)
        >>> for result in results:
        ...     print(result.format_output())
    """
    logger.info(f"Running {len(checkpoints)} checkpoints")
    logger.info("="*70)
    
    results = []
    
    for i, checkpoint in enumerate(checkpoints, 1):
        logger.info(f"\nCheckpoint {i}/{len(checkpoints)}: {checkpoint[:50]}...")
        
        result = evaluate_checkpoint(
            checkpoint=checkpoint,
            repo_path=repo_path,
            retriever=retriever,
            use_llm=use_llm,
            api_key=api_key,
            model_name=model_name
        )
        
        results.append(result)
        
        # Log result
        status = "✓ PASS" if result.passed else "✗ FAIL"
        logger.info(f"{status}: {result.explanation[:100]}")
        
        # Stop on failure if requested
        if stop_on_failure and not result.passed:
            logger.warning(f"Stopping on failure at checkpoint {i}")
            break
    
    # Summary
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    logger.info("\n" + "="*70)
    logger.info(f"Checkpoint Summary: {passed_count}/{total} passed")
    logger.info("="*70)
    
    return results


def format_results_summary(results: List[CheckpointResult]) -> str:
    """
    Format checkpoint results as a summary report.
    
    Args:
        results: List of CheckpointResult objects
    
    Returns:
        Formatted summary string
    """
    output = []
    output.append("="*70)
    output.append("CHECKPOINT VALIDATION RESULTS")
    output.append("="*70)
    output.append("")
    
    for i, result in enumerate(results, 1):
        output.append(f"{i}. {result.format_output()}")
    
    # Summary statistics
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    pass_rate = (passed / len(results) * 100) if results else 0
    
    output.append("="*70)
    output.append("SUMMARY")
    output.append("="*70)
    output.append(f"Total Checkpoints: {len(results)}")
    output.append(f"Passed: {passed}")
    output.append(f"Failed: {failed}")
    output.append(f"Pass Rate: {pass_rate:.1f}%")
    output.append("="*70)
    
    return "\n".join(output)
