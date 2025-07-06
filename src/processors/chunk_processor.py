"""
Chunk processor for extracting and validating chunks from LMM output.
"""

import re
import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Import Chunk from models
from ..models import Chunk

logger = logging.getLogger(__name__)


class ChunkProcessor:
    """
    Processes raw LMM output to extract and validate structured chunks.
    
    This component:
    1. Extracts chunk data from raw LMM output
    2. Validates chunk structure and metadata
    3. Post-processes chunks (e.g., merging related chunks)
    """
    
    def __init__(self):
        """Initialize the chunk processor."""
        # Regex patterns for extracting chunks and their components
        self.chunk_pattern = re.compile(
            r'\[CHUNK\](.*?)\[/CHUNK\]',
            re.DOTALL
        )
        self.heading_pattern = re.compile(
            r'\[HEADING_HIERARCHY\](.*?)\[/HEADING_HIERARCHY\]',
            re.DOTALL
        )
        self.content_pattern = re.compile(
            r'\[CONTENT\](.*?)\[/CONTENT\]',
            re.DOTALL
        )
        self.continuation_pattern = re.compile(
            r'\[CONTINUES\](.*?)\[/CONTINUES\]',
            re.DOTALL
        )
    
    def process(
        self,
        raw_lmm_output: str,
        batch_idx: int,
        page_numbers: List[int],
    ) -> List["Chunk"]:  # Use string annotation
        """
        Process raw LMM output to extract and validate chunks.
        
        Args:
            raw_lmm_output: Raw output from the LMM
            batch_idx: Index of the current batch
            page_numbers: List of page numbers in the current batch
            
        Returns:
            List of validated chunks
        """
        # Extract chunks from raw output
        chunk_matches = self.chunk_pattern.finditer(raw_lmm_output)
        chunks = []
        
        for i, match in enumerate(chunk_matches):
            chunk_content = match.group(1)
            
            # Extract components
            heading_match = self.heading_pattern.search(chunk_content)
            content_match = self.content_pattern.search(chunk_content)
            continuation_match = self.continuation_pattern.search(chunk_content)
            
            if not heading_match or not content_match:
                logger.warning(f"Chunk {i} in batch {batch_idx} is missing required components")
                continue
            
            # Extract values
            heading_hierarchy = [
                h.strip() for h in heading_match.group(1).split('>')
            ]
            content = content_match.group(1).strip()
            
            # Default continuation flag to "False" if not present
            continuation_flag = "False"
            if continuation_match:
                continuation_flag = continuation_match.group(1).strip()
                # Validate continuation flag
                if continuation_flag not in ["True", "False", "Partial"]:
                    logger.warning(
                        f"Invalid continuation flag in chunk {i}, batch {batch_idx}: {continuation_flag}"
                    )
                    continuation_flag = "False"
            
            # Create chunk
            chunk = Chunk(
                id=f"b{batch_idx}_c{i}_{uuid.uuid4().hex[:8]}",
                content=content,
                heading_hierarchy=heading_hierarchy,
                page_numbers=page_numbers,
                continuation_flag=continuation_flag,
                source_batch=batch_idx,
                metadata={
                    "position_in_batch": i,
                    "raw_length": len(content),
                    "heading_count": len(heading_hierarchy),
                }
            )
            
            chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from batch {batch_idx}")
        return chunks
    
    def post_process_all(self, chunks: List["Chunk"]) -> List["Chunk"]:
        """
        Post-process all chunks after all batches have been processed.
        
        This includes:
        1. Merging related chunks based on continuation flags
        2. Final validation and cleanup
        
        Args:
            chunks: List of all chunks from all batches
            
        Returns:
            List of post-processed chunks
        """
        if not chunks:
            return []
            
        # Sort chunks by batch and position
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (c.source_batch, c.metadata.get("position_in_batch", 0))
        )
        
        # Merge related chunks
        merged_chunks = []
        current_chunk = None
        
        for chunk in sorted_chunks:
            if current_chunk is None:
                # First chunk
                current_chunk = chunk
                continue
                
            if chunk.continuation_flag == "True":
                # Merge with previous chunk
                current_chunk = self._merge_chunks(current_chunk, chunk)
            else:
                # Save previous chunk and start a new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk
                
        # Don't forget the last chunk
        if current_chunk is not None:
            merged_chunks.append(current_chunk)
            
        logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
        
        # Final validation and cleanup
        return self._validate_final_chunks(merged_chunks)
    
    def _merge_chunks(self, chunk1: "Chunk", chunk2: "Chunk") -> "Chunk":
        """
        Merge two chunks, preserving important metadata.
        
        Args:
            chunk1: First chunk
            chunk2: Second chunk (continuation of the first)
            
        Returns:
            Merged chunk
        """
        # Combine content
        combined_content = f"{chunk1.content}\n\n{chunk2.content}"
        
        # Keep the heading hierarchy from the first chunk
        heading_hierarchy = chunk1.heading_hierarchy
        
        # Combine page numbers
        page_numbers = sorted(set(chunk1.page_numbers + chunk2.page_numbers))
        
        # Combine metadata
        metadata = chunk1.metadata.copy()
        metadata.update({
            "merged_from": [chunk1.id, chunk2.id],
            "original_continuation_flag": chunk2.continuation_flag,
        })
        
        # Create merged chunk
        return Chunk(
            id=f"merged_{chunk1.id}_{chunk2.id}",
            content=combined_content,
            heading_hierarchy=heading_hierarchy,
            page_numbers=page_numbers,
            continuation_flag="False",  # Reset continuation flag
            source_batch=chunk1.source_batch,  # Keep source batch from first chunk
            metadata=metadata,
        )
    
    def _validate_final_chunks(self, chunks: List["Chunk"]) -> List["Chunk"]:
        """
        Perform final validation and cleanup of chunks.
        
        Args:
            chunks: List of processed chunks
            
        Returns:
            List of validated chunks
        """
        valid_chunks = []
        
        for chunk in chunks:
            # Skip empty chunks
            if not chunk.content.strip():
                logger.warning(f"Skipping empty chunk: {chunk.id}")
                continue
                
            # Ensure heading hierarchy is valid
            if not chunk.heading_hierarchy:
                logger.warning(f"Chunk {chunk.id} has no heading hierarchy")
                # Assign a default heading
                chunk.heading_hierarchy = ["Unknown Section"]
                
            # Cleanup content (remove extra whitespace, etc.)
            chunk.content = self._cleanup_content(chunk.content)
            
            valid_chunks.append(chunk)
            
        return valid_chunks
    
    def _cleanup_content(self, content: str) -> str:
        """
        Clean up chunk content.
        
        Args:
            content: Raw chunk content
            
        Returns:
            Cleaned content
        """
        # Remove any remaining markdown or special tags
        content = re.sub(r'\[.*?\]', '', content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
