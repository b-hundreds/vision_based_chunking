"""
Vision-Based Chunking for RAG Systems
=====================================

This package implements an advanced document chunking algorithm for RAG systems
using Large Multimodal Models to understand both the visual layout and textual content.
"""

__version__ = "0.1.0"

# Expose the Chunk class
from .models import Chunk

# Avoid circular imports by deferring import
def get_vision_chunker():
    from .chunker import VisionChunker
    return VisionChunker

def get_simple_vision_chunker():
    from .simple_chunker import SimpleVisionChunker
    return SimpleVisionChunker

__all__ = ["get_vision_chunker", "get_simple_vision_chunker", "Chunk"]
