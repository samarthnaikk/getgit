"""
Advanced examples demonstrating RAG integration for Repository QnA and Review.

This script shows how the RAG system can be used to build:
1. Interactive Repository QnA system
2. Repository Review and Analysis system
"""

import os
from typing import List, Dict
from rag import (
    RepositoryChunker,
    SimpleEmbedding,
    Retriever,
    RAGConfig,
    ChunkType,
    RetrievalResult
)


class RepositoryQnA:
    """
    Interactive Question-Answering system for repositories.
    
    Uses RAG to retrieve relevant context and generate informed responses.
    """
    
    def __init__(self, repo_path: str, repo_name: str = ""):
        """Initialize QnA system with a repository."""
        self.repo_path = repo_path
        self.repo_name = repo_name or os.path.basename(repo_path)
        
        # Initialize RAG components
        config = RAGConfig.default()
        self.chunker = RepositoryChunker(repo_path, repo_name)
        self.embedding_model = SimpleEmbedding(max_features=384)
        self.retriever = Retriever(self.embedding_model)
        
        # Index the repository
        print(f"Indexing repository: {self.repo_name}...")
        chunks = self.chunker.chunk_repository(config.chunking.file_patterns)
        
        if len(chunks) == 0:
            raise ValueError(
                f"No chunks found in repository {repo_path}. "
                "Make sure the repository contains supported file types."
            )
        
        self.retriever.index_chunks(chunks)
        print(f"✓ Indexed {len(self.retriever)} chunks")
    
    def ask(self, question: str, top_k: int = 3) -> Dict:
        """
        Ask a question about the repository.
        
        Args:
            question: Natural language question
            top_k: Number of relevant contexts to retrieve
        
        Returns:
            Dictionary with question, context, and results
        """
        results = self.retriever.retrieve(question, top_k=top_k)
        
        # Build context from results
        context_parts = []
        for result in results:
            context_parts.append({
                'file': result.chunk.file_path,
                'type': result.chunk.chunk_type.value,
                'score': result.score,
                'content': result.chunk.content,
                'metadata': result.chunk.metadata
            })
        
        return {
            'question': question,
            'num_results': len(results),
            'contexts': context_parts
        }
    
    def format_answer(self, qna_result: Dict) -> str:
        """Format QnA result for display."""
        output = []
        output.append(f"\n{'='*70}")
        output.append(f"Question: {qna_result['question']}")
        output.append(f"{'='*70}")
        
        if qna_result['num_results'] == 0:
            output.append("\nNo relevant information found.")
            return '\n'.join(output)
        
        output.append(f"\nFound {qna_result['num_results']} relevant context(s):\n")
        
        for i, ctx in enumerate(qna_result['contexts'], 1):
            output.append(f"[{i}] {ctx['file']} (score: {ctx['score']:.4f})")
            output.append(f"    Type: {ctx['type']}")
            if ctx['metadata']:
                output.append(f"    Metadata: {ctx['metadata']}")
            
            # Show content preview
            content_lines = ctx['content'].split('\n')[:10]
            output.append(f"    Content preview:")
            for line in content_lines:
                output.append(f"      {line}")
            if len(ctx['content'].split('\n')) > 10:
                output.append(f"      ... ({len(ctx['content'].split('\n')) - 10} more lines)")
            output.append("")
        
        return '\n'.join(output)


class RepositoryReview:
    """
    Automated repository review and analysis system.
    
    Uses RAG to analyze code structure, documentation, and best practices.
    """
    
    def __init__(self, repo_path: str, repo_name: str = ""):
        """Initialize review system with a repository."""
        self.repo_path = repo_path
        self.repo_name = repo_name or os.path.basename(repo_path)
        
        # Chunk the repository
        config = RAGConfig.default()
        self.chunker = RepositoryChunker(repo_path, repo_name)
        self.chunks = self.chunker.chunk_repository(config.chunking.file_patterns)
    
    def generate_review(self) -> Dict:
        """
        Generate comprehensive repository review.
        
        Returns:
            Dictionary with review metrics and insights
        """
        review = {
            'repository': self.repo_name,
            'total_chunks': len(self.chunks),
            'structure': self._analyze_structure(),
            'code_analysis': self._analyze_code(),
            'documentation': self._analyze_documentation(),
            'recommendations': []
        }
        
        # Generate recommendations
        review['recommendations'] = self._generate_recommendations(review)
        
        return review
    
    def _analyze_structure(self) -> Dict:
        """Analyze repository structure."""
        files = set(chunk.file_path for chunk in self.chunks)
        file_types = {}
        
        for file in files:
            ext = os.path.splitext(file)[1] or 'no_extension'
            file_types[ext] = file_types.get(ext, 0) + 1
        
        return {
            'total_files': len(files),
            'file_types': file_types
        }
    
    def _analyze_code(self) -> Dict:
        """Analyze code chunks."""
        code_chunks = [
            c for c in self.chunks 
            if 'code' in c.chunk_type.value.lower()
        ]
        
        functions = [c for c in code_chunks if c.chunk_type == ChunkType.CODE_FUNCTION]
        classes = [c for c in code_chunks if c.chunk_type == ChunkType.CODE_CLASS]
        
        # Analyze function complexity (simple line count)
        avg_function_lines = 0
        if functions:
            avg_function_lines = sum(
                c.end_line - c.start_line for c in functions
            ) / len(functions)
        
        return {
            'total_code_chunks': len(code_chunks),
            'functions': len(functions),
            'classes': len(classes),
            'avg_function_lines': round(avg_function_lines, 1)
        }
    
    def _analyze_documentation(self) -> Dict:
        """Analyze documentation chunks."""
        doc_chunks = [
            c for c in self.chunks 
            if c.chunk_type == ChunkType.MARKDOWN_SECTION or 
               c.chunk_type == ChunkType.DOCUMENTATION
        ]
        
        readme_chunks = [c for c in doc_chunks if 'readme' in c.file_path.lower()]
        
        return {
            'total_doc_chunks': len(doc_chunks),
            'readme_sections': len(readme_chunks),
            'has_readme': len(readme_chunks) > 0
        }
    
    def _generate_recommendations(self, review: Dict) -> List[str]:
        """Generate recommendations based on review."""
        recommendations = []
        
        # Check documentation
        if not review['documentation']['has_readme']:
            recommendations.append(
                "❗ Add a README.md file to document the project"
            )
        elif review['documentation']['readme_sections'] < 5:
            recommendations.append(
                "ℹ️ Consider expanding README documentation with more sections"
            )
        
        # Check code structure
        if review['code_analysis']['functions'] == 0:
            recommendations.append(
                "ℹ️ No functions detected - consider organizing code into functions"
            )
        
        if review['code_analysis']['avg_function_lines'] > 50:
            recommendations.append(
                f"⚠️ Average function length is {review['code_analysis']['avg_function_lines']} lines. "
                "Consider breaking down large functions."
            )
        
        # Check file diversity
        if len(review['structure']['file_types']) == 1:
            recommendations.append(
                "ℹ️ Repository contains only one file type - consider adding documentation"
            )
        
        if not recommendations:
            recommendations.append("✅ Repository structure looks good!")
        
        return recommendations
    
    def format_review(self, review: Dict) -> str:
        """Format review for display."""
        output = []
        output.append(f"\n{'='*70}")
        output.append(f"Repository Review: {review['repository']}")
        output.append(f"{'='*70}\n")
        
        # Structure
        output.append("📁 Repository Structure")
        output.append(f"  Total Files: {review['structure']['total_files']}")
        output.append(f"  File Types:")
        for ext, count in sorted(review['structure']['file_types'].items()):
            output.append(f"    {ext}: {count} file(s)")
        
        # Code Analysis
        output.append(f"\n💻 Code Analysis")
        output.append(f"  Total Code Chunks: {review['code_analysis']['total_code_chunks']}")
        output.append(f"  Functions: {review['code_analysis']['functions']}")
        output.append(f"  Classes: {review['code_analysis']['classes']}")
        output.append(f"  Avg Function Length: {review['code_analysis']['avg_function_lines']} lines")
        
        # Documentation
        output.append(f"\n📝 Documentation")
        output.append(f"  Total Doc Chunks: {review['documentation']['total_doc_chunks']}")
        output.append(f"  README Sections: {review['documentation']['readme_sections']}")
        output.append(f"  Has README: {'Yes ✓' if review['documentation']['has_readme'] else 'No ✗'}")
        
        # Recommendations
        output.append(f"\n💡 Recommendations")
        for rec in review['recommendations']:
            output.append(f"  {rec}")
        
        output.append(f"\n{'='*70}")
        
        return '\n'.join(output)


def demo_qna():
    """Demonstrate Repository QnA system."""
    print("\n" + "="*70)
    print("DEMO: Repository Question & Answer System")
    print("="*70)
    
    # Use the cloned source_repo if available, otherwise use current directory
    repo_path = "source_repo" if os.path.exists("source_repo") else "."
    
    try:
        qna = RepositoryQnA(repo_path, repo_name="getgit")
        
        # Ask several questions
        questions = [
            "What is this project about?",
            "How do I use the RAG system?",
            "What are the main features?"
        ]
        
        for question in questions:
            result = qna.ask(question, top_k=2)
            print(qna.format_answer(result))
    except ValueError as e:
        print(f"\n⚠️ Could not initialize QnA system: {e}")
        print("Tip: Make sure you're running from a directory with supported files.")


def demo_review():
    """Demonstrate Repository Review system."""
    print("\n" + "="*70)
    print("DEMO: Repository Review System")
    print("="*70)
    
    # Use the cloned source_repo if available, otherwise use current directory
    repo_path = "source_repo" if os.path.exists("source_repo") else "."
    
    try:
        reviewer = RepositoryReview(repo_path, repo_name="getgit")
        
        if len(reviewer.chunks) == 0:
            print("\n⚠️ No chunks found in repository.")
            print("Tip: Make sure you're running from a directory with supported files.")
            return
        
        review = reviewer.generate_review()
        print(reviewer.format_review(review))
    except Exception as e:
        print(f"\n⚠️ Could not generate review: {e}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("Advanced RAG Integration Examples")
    print("="*70)
    
    # Demo 1: Repository QnA
    demo_qna()
    
    # Demo 2: Repository Review
    demo_review()
    
    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("="*70)


if __name__ == "__main__":
    main()
