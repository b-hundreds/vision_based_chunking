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
        
        # Count to track chunk extraction for logging
        match_count = 0
        valid_chunk_count = 0
        
        # Track unique heading hierarchies in this batch for validation
        batch_hierarchies = {}
        
        for i, match in enumerate(chunk_matches):
            match_count += 1
            chunk_content = match.group(1)
            
            # Extract components
            heading_match = self.heading_pattern.search(chunk_content)
            content_match = self.content_pattern.search(chunk_content)
            continuation_match = self.continuation_pattern.search(chunk_content)
            
            if not heading_match or not content_match:
                logger.warning(f"Chunk {i} in batch {batch_idx} is missing required components")
                if not heading_match:
                    logger.debug(f"Missing heading hierarchy in batch {batch_idx}, chunk {i}")
                if not content_match:
                    logger.debug(f"Missing content in batch {batch_idx}, chunk {i}")
                
                # Try to salvage chunks with missing components by assigning defaults
                heading_hierarchy = ["Unknown Section"]
                if heading_match:
                    heading_hierarchy = [h.strip() for h in heading_match.group(1).split('>')]
                    
                content = "No content extracted"
                if content_match:
                    content = content_match.group(1).strip()
                    
                # Only continue if we have at least some content
                if content == "No content extracted" or len(content.strip()) < 10:
                    logger.warning(f"Skipping chunk with insufficient content in batch {batch_idx}")
                    continue
            else:
                # Extract values
                heading_hierarchy = [h.strip() for h in heading_match.group(1).split('>')]
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
            
            # Track hierarchies to detect potential mismatches
            hierarchy_key = ">".join(heading_hierarchy)
            if hierarchy_key not in batch_hierarchies:
                batch_hierarchies[hierarchy_key] = []
            batch_hierarchies[hierarchy_key].append(i)
                
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
            valid_chunk_count += 1
        
        # Log warning if we detect many chunks with the same heading hierarchy
        # This might indicate improper heading detection
        if len(batch_hierarchies) == 1 and len(chunks) > 3:
            hierarchy = list(batch_hierarchies.keys())[0]
            logger.warning(
                f"Batch {batch_idx} (pages {page_numbers}) has {len(chunks)} chunks "
                f"all with the same heading hierarchy: {hierarchy}"
            )
        
        # If no chunks were found but there was output from the LMM, try a fallback approach
        if match_count == 0 and len(raw_lmm_output) > 100:
            logger.warning(f"No chunk patterns found in batch {batch_idx}, trying fallback chunk extraction")
            # Look for any content that might be useful - this is a very simple fallback
            fallback_chunk = self._extract_fallback_chunk(raw_lmm_output, batch_idx, page_numbers)
            if fallback_chunk:
                chunks.append(fallback_chunk)
                valid_chunk_count += 1
        
        logger.info(f"Extracted {len(chunks)} chunks from batch {batch_idx} (found {match_count} chunk patterns)")
        return chunks
        
    def _extract_fallback_chunk(self, raw_output: str, batch_idx: int, page_numbers: List[int]) -> Optional[Chunk]:
        """
        Extract a fallback chunk when normal pattern matching fails.
        
        Args:
            raw_output: Raw LMM output
            batch_idx: Batch index
            page_numbers: List of page numbers
            
        Returns:
            A Chunk object or None if no usable content found
        """
        # Remove any markdown formatting to get plain text
        cleaned_text = re.sub(r'#+ ', '', raw_output)  # Remove headings
        cleaned_text = re.sub(r'\*\*|\*|__|\||```', '', cleaned_text)  # Remove bold, italic, code blocks
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        
        # Filter out short paragraphs and join the rest
        content = '\n\n'.join([p for p in paragraphs if len(p) > 30])
        
        if len(content) < 100:
            logger.warning(f"Fallback content too short in batch {batch_idx}: {len(content)} chars")
            return None
            
        logger.info(f"Created fallback chunk with {len(content)} chars for batch {batch_idx}")
        
        # Create a fallback chunk
        return Chunk(
            id=f"fallback_b{batch_idx}_{uuid.uuid4().hex[:8]}",
            content=content,
            heading_hierarchy=["Fallback Content"],
            page_numbers=page_numbers,
            continuation_flag="False",
            source_batch=batch_idx,
            metadata={
                "is_fallback": True,
                "raw_length": len(content),
            }
        )
    
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
        
        # First pass - check heading hierarchies against page numbers to detect potential issues
        page_to_heading_map = {}
        
        # Build a mapping of page numbers to heading hierarchies
        for chunk in chunks:
            for page in chunk.page_numbers:
                if page not in page_to_heading_map:
                    page_to_heading_map[page] = []
                page_to_heading_map[page].append(" > ".join(chunk.heading_hierarchy))
        
        # Build a mapping of headings to their page ranges
        heading_to_pages = {}
        for page, headings in page_to_heading_map.items():
            for heading in headings:
                if heading not in heading_to_pages:
                    heading_to_pages[heading] = []
                heading_to_pages[heading].append(page)
        
        # Check for heading assignments that span non-consecutive pages
        suspicious_headings = {}
        for heading, pages in heading_to_pages.items():
            pages.sort()
            page_ranges = []
            current_range = [pages[0]]
            
            for i in range(1, len(pages)):
                if pages[i] == pages[i-1] + 1:
                    current_range.append(pages[i])
                else:
                    page_ranges.append(current_range)
                    current_range = [pages[i]]
            
            if current_range:
                page_ranges.append(current_range)
            
            if len(page_ranges) > 1:
                range_strs = [f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in page_ranges]
                suspicious_headings[heading] = range_strs
        
        # Log any suspicious heading assignments
        if suspicious_headings:
            for heading, ranges in suspicious_headings.items():
                logger.warning(
                    f"Suspicious heading assignment: '{heading}' appears on non-consecutive page ranges: {', '.join(ranges)}"
                )
        
        # Second pass - clean up and validate each chunk
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
            
            # Check if this chunk's heading is in the suspicious list
            chunk_heading = " > ".join(chunk.heading_hierarchy)
            if chunk_heading in suspicious_headings:
                # Add a note to the metadata
                if "validation_warnings" not in chunk.metadata:
                    chunk.metadata["validation_warnings"] = []
                chunk.metadata["validation_warnings"].append(
                    f"This chunk's heading appears on non-consecutive page ranges: {', '.join(suspicious_headings[chunk_heading])}"
                )
                
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
