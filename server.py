"""
GetGit Flask Server - Single Entry Point
This module provides the Flask web interface for GetGit.
All business logic is delegated to core.py.
"""

from flask import Flask, render_template, request, jsonify
import logging
import os
from typing import Optional
import threading

# Import core module functions
from core import (
    initialize_repository,
    setup_rag,
    answer_query,
    validate_checkpoints,
    setup_logging as setup_core_logging
)
from rag import RAGConfig

# Configure Flask app
app = Flask(__name__)

# Configure Flask secret key for sessions
# In production, FLASK_SECRET_KEY environment variable must be set
# For development, generate a random key if not provided
import secrets
default_secret = secrets.token_hex(32) if os.environ.get('FLASK_ENV') == 'development' else None
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', default_secret)

if app.config['SECRET_KEY'] is None:
    raise ValueError("FLASK_SECRET_KEY environment variable must be set in production")

# Configure server logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('getgit.server')

# Global state to store retriever (in production, use Redis or similar)
# This is a simple in-memory storage for demo purposes
app_state = {
    'retriever': None,
    'repo_path': None,
    'repo_url': None
}

# Thread lock for thread-safe state access
state_lock = threading.Lock()


@app.route('/', methods=['GET'])
def home():
    """
    Render the home page.
    """
    return render_template('index.html')


@app.route('/initialize', methods=['POST'])
def initialize():
    """
    Initialize repository and setup RAG pipeline.
    
    Expected JSON payload:
    {
        "repo_url": "https://github.com/user/repo.git"
    }
    
    Returns:
    {
        "success": true/false,
        "message": "...",
        "repo_path": "...",
        "chunks_count": 123
    }
    """
    logger.info("Received repository initialization request")
    
    try:
        data = request.get_json()
        if not data or 'repo_url' not in data:
            logger.warning("Missing repo_url in request")
            return jsonify({
                'success': False,
                'message': 'Missing repo_url parameter'
            }), 400
        
        repo_url = data['repo_url'].strip()
        logger.info(f"Initializing repository: {repo_url}")
        
        # Step 1: Initialize repository
        repo_path = initialize_repository(repo_url, local_path="source_repo")
        logger.info(f"Repository initialized at {repo_path}")
        
        # Step 2: Setup RAG pipeline
        logger.info("Setting up RAG pipeline...")
        retriever = setup_rag(repo_path, repository_name=None, config=None)
        chunks_count = len(retriever)
        logger.info(f"RAG pipeline ready with {chunks_count} chunks")
        
        # Store in app state (repository-level persistence)
        with state_lock:
            app_state['retriever'] = retriever
            app_state['repo_path'] = repo_path
            app_state['repo_url'] = repo_url
        
        logger.info("Repository initialization completed successfully")
        return jsonify({
            'success': True,
            'message': f'Repository initialized successfully with {chunks_count} chunks',
            'repo_path': repo_path,
            'chunks_count': chunks_count
        })
    
    except Exception as e:
        logger.error(f"Repository initialization failed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error initializing repository: {str(e)}'
        }), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Answer a question about the repository using RAG + LLM.
    
    Expected JSON payload:
    {
        "query": "What is this project about?",
        "use_llm": true/false
    }
    
    Returns:
    {
        "success": true/false,
        "query": "...",
        "response": "...",
        "retrieved_chunks": [...],
        "error": "..." (if any)
    }
    """
    logger.info("Received question answering request")
    
    try:
        # Check if repository is initialized
        with state_lock:
            retriever = app_state['retriever']
        
        if retriever is None:
            logger.warning("Question asked without initializing repository")
            return jsonify({
                'success': False,
                'message': 'Repository not initialized. Please initialize a repository first.'
            }), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            logger.warning("Missing query in request")
            return jsonify({
                'success': False,
                'message': 'Missing query parameter'
            }), 400
        
        query = data['query'].strip()
        use_llm = data.get('use_llm', True)
        
        logger.info(f"Processing query: '{query}' (use_llm={use_llm})")
        
        # Process query using core.py
        result = answer_query(
            query=query,
            retriever=retriever,
            top_k=5,
            use_llm=use_llm
        )
        
        logger.info("Query processed successfully")
        
        return jsonify({
            'success': True,
            'query': result['query'],
            'response': result['response'],
            'retrieved_chunks': result['retrieved_chunks'],
            'context': result['context'],
            'error': result['error']
        })
    
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error processing query: {str(e)}'
        }), 500


@app.route('/checkpoints', methods=['POST'])
def run_checkpoints():
    """
    Run checkpoint validation on the initialized repository.
    
    Expected JSON payload:
    {
        "checkpoints_file": "checkpoints.txt" (optional, defaults to "checkpoints.txt"),
        "use_llm": true/false (optional, defaults to true)
    }
    
    Returns:
    {
        "success": true/false,
        "checkpoints": [...],
        "results": [...],
        "summary": "...",
        "passed_count": 3,
        "total_count": 5,
        "pass_rate": 60.0
    }
    """
    logger.info("Received checkpoint validation request")
    
    try:
        # Check if repository is initialized
        with state_lock:
            repo_url = app_state['repo_url']
            repo_path = app_state['repo_path']
        
        if repo_url is None:
            logger.warning("Checkpoints requested without initializing repository")
            return jsonify({
                'success': False,
                'message': 'Repository not initialized. Please initialize a repository first.'
            }), 400
        
        data = request.get_json() or {}
        checkpoints_file = data.get('checkpoints_file', 'checkpoints.txt')
        use_llm = data.get('use_llm', True)
        
        logger.info(f"Running checkpoints from {checkpoints_file} (use_llm={use_llm})")
        
        # Run checkpoint validation
        result = validate_checkpoints(
            repo_url=repo_url,
            checkpoints_file=checkpoints_file,
            local_path=repo_path,
            use_llm=use_llm,
            log_level='INFO'
        )
        
        # Convert CheckpointResult objects to dictionaries
        results_dict = [
            {
                'checkpoint': r.checkpoint,
                'passed': r.passed,
                'explanation': r.explanation,
                'evidence': r.evidence,
                'score': r.score
            }
            for r in result['results']
        ]
        
        logger.info(f"Checkpoint validation completed: {result['passed_count']}/{result['total_count']} passed")
        
        return jsonify({
            'success': True,
            'checkpoints': result['checkpoints'],
            'results': results_dict,
            'summary': result['summary'],
            'passed_count': result['passed_count'],
            'total_count': result['total_count'],
            'pass_rate': result['pass_rate']
        })
    
    except Exception as e:
        logger.error(f"Checkpoint validation failed: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error running checkpoints: {str(e)}'
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """
    Get the current status of the application.
    
    Returns:
    {
        "initialized": true/false,
        "repo_url": "..." (if initialized),
        "chunks_count": 123 (if initialized)
    }
    """
    with state_lock:
        is_initialized = app_state['retriever'] is not None
        
        response = {
            'initialized': is_initialized
        }
        
        if is_initialized:
            response['repo_url'] = app_state['repo_url']
            response['chunks_count'] = len(app_state['retriever'])
    
    return jsonify(response)


@app.route('/checkpoints/list', methods=['GET'])
def list_checkpoints():
    """
    Get all checkpoints from checkpoints.txt.
    
    Returns:
    {
        "success": true/false,
        "checkpoints": [...],
        "message": "..." (if error)
    }
    """
    logger.info("Received request to list checkpoints")
    
    try:
        checkpoints_file = 'checkpoints.txt'
        
        if not os.path.exists(checkpoints_file):
            return jsonify({
                'success': False,
                'checkpoints': [],
                'message': 'Checkpoints file not found'
            })
        
        with open(checkpoints_file, 'r') as f:
            lines = f.readlines()
        
        # Filter out empty lines and comments, clean up numbering
        checkpoints = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering if present (e.g., "1. " or "1) ")
                import re
                cleaned = re.sub(r'^\d+[\.\)]\s*', '', line)
                checkpoints.append(cleaned)
        
        logger.info(f"Retrieved {len(checkpoints)} checkpoints")
        return jsonify({
            'success': True,
            'checkpoints': checkpoints
        })
    
    except Exception as e:
        logger.error(f"Failed to list checkpoints: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'checkpoints': [],
            'message': f'Error reading checkpoints: {str(e)}'
        }), 500


@app.route('/checkpoints/add', methods=['POST'])
def add_checkpoint():
    """
    Add a new checkpoint to checkpoints.txt.
    
    Expected JSON payload:
    {
        "checkpoint": "Check if the repository has tests"
    }
    
    Returns:
    {
        "success": true/false,
        "message": "...",
        "checkpoints": [...] (updated list)
    }
    """
    logger.info("Received request to add checkpoint")
    
    try:
        data = request.get_json()
        if not data or 'checkpoint' not in data:
            logger.warning("Missing checkpoint in request")
            return jsonify({
                'success': False,
                'message': 'Missing checkpoint parameter'
            }), 400
        
        checkpoint = data['checkpoint'].strip()
        if not checkpoint:
            return jsonify({
                'success': False,
                'message': 'Checkpoint cannot be empty'
            }), 400
        
        checkpoints_file = 'checkpoints.txt'
        
        # Read existing checkpoints to get count
        existing_checkpoints = []
        if os.path.exists(checkpoints_file):
            with open(checkpoints_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    existing_checkpoints.append(line)
        
        # Append new checkpoint with numbering
        next_number = len(existing_checkpoints) + 1
        with open(checkpoints_file, 'a') as f:
            f.write(f"{next_number}. {checkpoint}\n")
        
        logger.info(f"Added checkpoint: {checkpoint}")
        
        # Return updated list
        existing_checkpoints.append(f"{next_number}. {checkpoint}")
        return jsonify({
            'success': True,
            'message': 'Checkpoint added successfully',
            'checkpoints': existing_checkpoints
        })
    
    except Exception as e:
        logger.error(f"Failed to add checkpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error adding checkpoint: {str(e)}'
        }), 500


if __name__ == '__main__':
    logger.info("="*70)
    logger.info("GetGit Server Starting")
    logger.info("Single entry point for repository analysis")
    logger.info("="*70)
    
    # Debug mode should only be enabled in development
    # Set FLASK_ENV=development to enable debug mode
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    # Port can be configured via environment variable, defaults to 5000
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
