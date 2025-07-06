"""
Shared data models for the chunking system.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """A single chunk of document content with metadata."""
    
    id: str
    content: str
    heading_hierarchy: List[str]
    page_numbers: List[int]
    continuation_flag: str  # 'True', 'False', or 'Partial'
    source_batch: int
    metadata: Dict[str, Any]
