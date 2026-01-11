"""
Repository persistence and validation module.

This module handles:
- Storing and retrieving the currently indexed repository URL
- Detecting repository changes
- Cleaning up old repository data when a new repository is provided
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger('getgit.repo_manager')


class RepositoryManager:
    """Manages repository persistence and cleanup."""
    
    def __init__(self, data_dir: str = "data", repo_dir: str = "source_repo", 
                 cache_dir: str = ".rag_cache"):
        """
        Initialize the repository manager.
        
        Args:
            data_dir: Directory to store persistence data
            repo_dir: Directory where repositories are cloned
            cache_dir: Directory for vector store cache
        """
        self.data_dir = Path(data_dir)
        self.repo_dir = Path(repo_dir)
        self.cache_dir = Path(cache_dir)
        self.source_file = self.data_dir / "source_repo.txt"
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_current_repo_url(self) -> Optional[str]:
        """
        Get the currently indexed repository URL.
        
        Returns:
            The repository URL if found, None otherwise
        """
        if not self.source_file.exists():
            logger.debug("No source_repo.txt found")
            return None
        
        try:
            with open(self.source_file, 'r') as f:
                url = f.read().strip()
            logger.info(f"Current repository URL: {url}")
            return url if url else None
        except Exception as e:
            logger.error(f"Error reading source_repo.txt: {e}")
            return None
    
    def set_current_repo_url(self, repo_url: str) -> None:
        """
        Store the current repository URL.
        
        Args:
            repo_url: The repository URL to store
        """
        try:
            with open(self.source_file, 'w') as f:
                f.write(repo_url.strip())
            logger.info(f"Stored repository URL: {repo_url}")
        except Exception as e:
            logger.error(f"Error writing source_repo.txt: {e}")
            raise
    
    def needs_reset(self, new_repo_url: str) -> bool:
        """
        Check if the repository needs to be reset.
        
        Args:
            new_repo_url: The new repository URL to check
        
        Returns:
            True if reset is needed, False otherwise
        """
        current_url = self.get_current_repo_url()
        
        if current_url is None:
            logger.info("No current repository, reset not needed")
            return False
        
        needs_reset = current_url.strip() != new_repo_url.strip()
        if needs_reset:
            logger.info(f"Repository URL changed from '{current_url}' to '{new_repo_url}'")
        else:
            logger.info("Repository URL unchanged")
        
        return needs_reset
    
    def cleanup(self) -> None:
        """
        Clean up all repository data.
        
        Removes:
        - Repository directory
        - Vector store cache
        - Embeddings
        """
        logger.info("Starting repository cleanup...")
        
        # Remove repository directory
        if self.repo_dir.exists():
            try:
                shutil.rmtree(self.repo_dir)
                logger.info(f"Deleted repository directory: {self.repo_dir}")
            except Exception as e:
                logger.error(f"Error deleting repository directory: {e}")
                raise
        
        # Remove cache directory
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Deleted cache directory: {self.cache_dir}")
            except Exception as e:
                logger.error(f"Error deleting cache directory: {e}")
                raise
        
        logger.info("Repository cleanup completed")
    
    def prepare_for_new_repo(self, repo_url: str) -> bool:
        """
        Prepare for a new repository by cleaning up if needed.
        
        Args:
            repo_url: The new repository URL
        
        Returns:
            True if cleanup was performed, False if reusing existing
        """
        if self.needs_reset(repo_url):
            logger.info("Repository change detected, performing cleanup...")
            self.cleanup()
            self.set_current_repo_url(repo_url)
            return True
        else:
            # Even if URL hasn't changed, store it if it's the first time
            if self.get_current_repo_url() is None:
                self.set_current_repo_url(repo_url)
            return False
