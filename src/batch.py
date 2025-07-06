"""
Batch processing module for handling document pages in batches.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class PageImage:
    """A single document page as an image with metadata."""
    
    image: Any  # PIL.Image or bytes
    page_number: int
    width: int
    height: int
    dpi: int = 300


class BatchProcessor:
    """
    Creates batches of document pages for processing.
    
    This component splits document pages into batches of a specified size
    to enable efficient processing while preserving context across pages.
    """
    
    def __init__(self, batch_size: int = 4):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Number of pages per batch (default: 4)
        """
        self.batch_size = batch_size
    
    def create_batches(self, pages: List[PageImage]) -> List[List[PageImage]]:
        """
        Split pages into batches of the specified size.
        
        Args:
            pages: List of page images
            
        Returns:
            List of batches, where each batch is a list of page images
        """
        batches = []
        
        for i in range(0, len(pages), self.batch_size):
            batch = pages[i:i+self.batch_size]
            batches.append(batch)
            
        return batches
