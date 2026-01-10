"""
Chunking strategies for repository content.

Provides intelligent chunking of source code, documentation, and configuration files
into semantically meaningful units for embedding and retrieval.
"""

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class ChunkType(Enum):
    """Types of chunks based on content."""
    CODE_FUNCTION = "code_function"
    CODE_CLASS = "code_class"
    CODE_METHOD = "code_method"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    MARKDOWN_SECTION = "markdown_section"
    COMMIT_MESSAGE = "commit_message"
    GENERIC = "generic"


@dataclass
class Chunk:
    """
    Represents a semantic chunk of repository content.
    
    Attributes:
        content: The actual text content of the chunk
        chunk_type: Type of chunk (function, class, documentation, etc.)
        file_path: Relative path to the file in the repository
        start_line: Starting line number in the file (1-indexed)
        end_line: Ending line number in the file (1-indexed)
        metadata: Additional metadata (e.g., function name, class name)
        repository: Repository identifier/name
    """
    content: str
    chunk_type: ChunkType
    file_path: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any]
    repository: str = ""
    
    def __repr__(self):
        return (f"Chunk(type={self.chunk_type.value}, file={self.file_path}, "
                f"lines={self.start_line}-{self.end_line})")


class RepositoryChunker:
    """
    Main chunker class for processing repository content.
    
    Supports multiple file types and chunking strategies tailored for code
    and documentation analysis.
    """
    
    def __init__(self, repository_path: str, repository_name: str = ""):
        """
        Initialize the chunker with a repository path.
        
        Args:
            repository_path: Path to the cloned repository
            repository_name: Name/identifier for the repository
        """
        self.repository_path = repository_path
        self.repository_name = repository_name or os.path.basename(repository_path)
        
    def chunk_repository(self, file_patterns: Optional[List[str]] = None) -> List[Chunk]:
        """
        Chunk entire repository based on file patterns.
        
        Args:
            file_patterns: List of glob patterns to include (e.g., ['*.py', '*.md'])
                          If None, processes all supported file types
        
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        # Default patterns if none provided
        if file_patterns is None:
            file_patterns = ['*.py', '*.md', '*.txt', '*.json', '*.yaml', '*.yml']
        
        for root, _, files in os.walk(self.repository_path):
            # Skip hidden directories and common exclusions
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue
            if any(excl in root for excl in ['__pycache__', 'node_modules', '.git']):
                continue
                
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.repository_path)
                
                # Check if file matches patterns
                if not self._matches_patterns(file, file_patterns):
                    continue
                
                try:
                    file_chunks = self.chunk_file(file_path, rel_path)
                    chunks.extend(file_chunks)
                except Exception as e:
                    # Log error but continue processing
                    print(f"Warning: Could not chunk file {rel_path}: {e}")
        
        return chunks
    
    def chunk_file(self, file_path: str, relative_path: str) -> List[Chunk]:
        """
        Chunk a single file based on its type.
        
        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from repository root
        
        Returns:
            List of Chunk objects for the file
        """
        extension = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return []
        
        if extension == '.py':
            return self._chunk_python_file(content, relative_path)
        elif extension == '.md':
            return self._chunk_markdown_file(content, relative_path)
        elif extension in ['.json', '.yaml', '.yml']:
            return self._chunk_config_file(content, relative_path, extension)
        else:
            return self._chunk_generic_file(content, relative_path)
    
    def _chunk_python_file(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk Python file into functions and classes.
        
        Uses regex-based parsing for simplicity. For production use,
        consider using ast module for more robust parsing.
        """
        chunks = []
        lines = content.split('\n')
        
        # Pattern for class definitions
        class_pattern = re.compile(r'^class\s+(\w+).*:')
        # Pattern for function/method definitions
        func_pattern = re.compile(r'^(\s*)def\s+(\w+)\s*\(')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                start_line = i + 1  # 1-indexed
                
                # Find end of class (next class or function at same indent level)
                indent = len(line) - len(line.lstrip())
                end_line = self._find_block_end(lines, i, indent)
                
                chunk_content = '\n'.join(lines[i:end_line])
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE_CLASS,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    metadata={'class_name': class_name},
                    repository=self.repository_name
                ))
                i = end_line
                continue
            
            # Check for function definition
            func_match = func_pattern.match(line)
            if func_match:
                func_name = func_match.group(2)
                indent = len(func_match.group(1))
                start_line = i + 1  # 1-indexed
                
                # Find end of function
                end_line = self._find_block_end(lines, i, indent)
                
                chunk_content = '\n'.join(lines[i:end_line])
                chunks.append(Chunk(
                    content=chunk_content,
                    chunk_type=ChunkType.CODE_FUNCTION,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    metadata={'function_name': func_name},
                    repository=self.repository_name
                ))
                i = end_line
                continue
            
            i += 1
        
        # If no functions/classes found, treat as generic
        if not chunks:
            chunks.append(Chunk(
                content=content,
                chunk_type=ChunkType.GENERIC,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                metadata={},
                repository=self.repository_name
            ))
        
        return chunks
    
    def _chunk_markdown_file(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk Markdown file by sections (headers).
        """
        chunks = []
        lines = content.split('\n')
        
        # Pattern for markdown headers
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        current_section = []
        current_start = 1
        current_header = None
        current_level = 0
        
        for i, line in enumerate(lines):
            header_match = header_pattern.match(line)
            
            if header_match:
                # Save previous section if exists
                if current_section:
                    chunks.append(Chunk(
                        content='\n'.join(current_section),
                        chunk_type=ChunkType.MARKDOWN_SECTION,
                        file_path=file_path,
                        start_line=current_start,
                        end_line=i,
                        metadata={'header': current_header, 'level': current_level},
                        repository=self.repository_name
                    ))
                
                # Start new section
                current_level = len(header_match.group(1))
                current_header = header_match.group(2)
                current_section = [line]
                current_start = i + 1  # 1-indexed
            else:
                current_section.append(line)
        
        # Add last section
        if current_section:
            chunks.append(Chunk(
                content='\n'.join(current_section),
                chunk_type=ChunkType.MARKDOWN_SECTION,
                file_path=file_path,
                start_line=current_start,
                end_line=len(lines),
                metadata={'header': current_header, 'level': current_level},
                repository=self.repository_name
            ))
        
        return chunks
    
    def _chunk_config_file(self, content: str, file_path: str, 
                          extension: str) -> List[Chunk]:
        """
        Chunk configuration files.
        
        For simplicity, treats entire config file as single chunk.
        Could be enhanced to parse JSON/YAML structure.
        """
        lines = content.split('\n')
        return [Chunk(
            content=content,
            chunk_type=ChunkType.CONFIGURATION,
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            metadata={'format': extension},
            repository=self.repository_name
        )]
    
    def _chunk_generic_file(self, content: str, file_path: str) -> List[Chunk]:
        """
        Chunk generic text files into fixed-size chunks with overlap.
        """
        chunks = []
        lines = content.split('\n')
        
        # For generic files, use line-based chunking
        chunk_size = 50  # lines per chunk
        overlap = 10     # lines of overlap
        
        i = 0
        while i < len(lines):
            end = min(i + chunk_size, len(lines))
            chunk_lines = lines[i:end]
            
            chunks.append(Chunk(
                content='\n'.join(chunk_lines),
                chunk_type=ChunkType.GENERIC,
                file_path=file_path,
                start_line=i + 1,  # 1-indexed
                end_line=end,
                metadata={},
                repository=self.repository_name
            ))
            
            i += chunk_size - overlap
        
        return chunks
    
    def _find_block_end(self, lines: List[str], start_idx: int, 
                        base_indent: int) -> int:
        """
        Find the end of a Python code block (class or function).
        
        Args:
            lines: All lines in the file
            start_idx: Starting index of the block
            base_indent: Base indentation level
        
        Returns:
            End index (exclusive)
        """
        i = start_idx + 1
        
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines and comments
            if not line.strip() or line.strip().startswith('#'):
                i += 1
                continue
            
            # Check indentation
            indent = len(line) - len(line.lstrip())
            
            # If we find a line at same or lower indent, block ends
            if indent <= base_indent:
                return i
            
            i += 1
        
        return len(lines)
    
    def _matches_patterns(self, filename: str, patterns: List[str]) -> bool:
        """
        Check if filename matches any of the given patterns.
        
        Args:
            filename: Name of the file
            patterns: List of glob-style patterns (e.g., '*.py')
        
        Returns:
            True if filename matches any pattern
        """
        import fnmatch
        return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)
