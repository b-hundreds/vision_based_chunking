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
        
        # Format patterns from prompt.txt
        self.format_continuation_pattern = re.compile(
            r'\[CONTINUES\](.*?)\[/CONTINUES\]',
            re.DOTALL
        )
        self.format_heading_pattern = re.compile(
            r'\[HEAD\](.*?)\[/HEAD\]',
            re.DOTALL
        )
        
        # Alternative patterns for different formatting
        self.alt_heading_pattern = re.compile(
            r'Heading Hierarchy:\s*(.*?)(?:\n|$)',
            re.DOTALL | re.IGNORECASE
        )
        self.alt_content_pattern = re.compile(
            r'Content:\s*(.*?)(?=\n\s*(?:Continues:|$))',
            re.DOTALL | re.IGNORECASE
        )
        self.alt_continuation_pattern = re.compile(
            r'Continues:\s*(True|False|Partial)',
            re.IGNORECASE
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
        # First try to extract chunks using the [CHUNK] tags (legacy format)
        chunk_matches = self.chunk_pattern.finditer(raw_lmm_output)
        chunks = []
        chunk_found = False
        
        # Try primary extraction method first ([CHUNK] tags)
        for i, match in enumerate(chunk_matches):
            chunk_found = True
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
        
        # If no chunks found with [CHUNK] tags, try the format from prompt.txt
        if not chunk_found:
            # Try to extract chunks using the format specified in prompt.txt
            chunks_from_format = self._extract_chunks_from_format(raw_lmm_output, batch_idx, page_numbers)
            if chunks_from_format:
                return chunks_from_format
            
            # If still no chunks found, try alternative extraction
            chunks = self._extract_chunks_alternative(raw_lmm_output, batch_idx, page_numbers)
            
        logger.info(f"Extracted {len(chunks)} chunks from batch {batch_idx}")
        return chunks
    
    def _extract_chunks_from_format(
        self,
        raw_lmm_output: str,
        batch_idx: int,
        page_numbers: List[int],
    ) -> List["Chunk"]:
        """
        Extract chunks using the format specified in prompt.txt.
        
        Format:
        [CONTINUES]True|False|Partial[/CONTINUES]
        [HEAD]main_heading > section_heading > chunk_heading[/HEAD]
        chunk_content
        
        Args:
            raw_lmm_output: Raw output from the LMM
            batch_idx: Index of the current batch
            page_numbers: List of page numbers in the current batch
            
        Returns:
            List of extracted chunks
        """
        chunks = []
        
        # Split the text by double newlines to find potential chunks
        sections = re.split(r'\n\s*\n', raw_lmm_output)
        
        # Look for patterns of [CONTINUES]...[/CONTINUES] followed by [HEAD]...[/HEAD]
        chunk_idx = 0
        for i in range(len(sections)):
            section = sections[i]
            
            # Look for continuation flag
            continuation_match = self.format_continuation_pattern.search(section)
            if not continuation_match:
                continue
                
            # Look for heading
            heading_match = self.format_heading_pattern.search(section)
            if not heading_match:
                continue
                
            # Extract values
            continuation_flag = continuation_match.group(1).strip()
            heading_text = heading_match.group(1).strip()
            heading_hierarchy = [h.strip() for h in heading_text.split('>')]
            
            # Validate continuation flag
            if continuation_flag not in ["True", "False", "Partial"]:
                logger.warning(
                    f"Invalid continuation flag in chunk {chunk_idx}, batch {batch_idx}: {continuation_flag}"
                )
                continuation_flag = "False"
                
            # Extract content - everything after the heading that's not the continuation flag
            content_text = section
            content_text = re.sub(r'\[CONTINUES\].*?\[/CONTINUES\]', '', content_text)
            content_text = re.sub(r'\[HEAD\].*?\[/HEAD\]', '', content_text)
            content_text = content_text.strip()
            
            # If there's no content in this section, look at the next section
            if not content_text and i + 1 < len(sections):
                content_text = sections[i + 1].strip()
                
            if content_text:
                # Create chunk
                chunk = Chunk(
                    id=f"b{batch_idx}_c{chunk_idx}_{uuid.uuid4().hex[:8]}",
                    content=content_text,
                    heading_hierarchy=heading_hierarchy,
                    page_numbers=page_numbers,
                    continuation_flag=continuation_flag,
                    source_batch=batch_idx,
                    metadata={
                        "position_in_batch": chunk_idx,
                        "raw_length": len(content_text),
                        "heading_count": len(heading_hierarchy),
                        "extraction_method": "format",
                    }
                )
                
                chunks.append(chunk)
                chunk_idx += 1
                
        return chunks
    
    def _extract_chunks_alternative(
        self,
        raw_lmm_output: str,
        batch_idx: int,
        page_numbers: List[int],
    ) -> List["Chunk"]:
        """
        Alternative method to extract chunks from LMM output that doesn't use the explicit [CHUNK] tags.
        
        Args:
            raw_lmm_output: Raw output from the LMM
            batch_idx: Index of the current batch
            page_numbers: List of page numbers in the current batch
            
        Returns:
            List of extracted chunks
        """
        chunks = []
        
        # Try to find chunks separated by headings and newlines
        # First split the text by double newlines to separate potential chunks
        lines = raw_lmm_output.split("\n")
        
        i = 0
        while i < len(lines):
            current_chunk = []
            heading_hierarchy = None
            content = ""
            continuation_flag = "False"
            
            # Look for a heading line
            while i < len(lines) and not heading_hierarchy:
                line = lines[i].strip()
                
                # Check if line looks like a heading hierarchy
                if ">" in line or "heading" in line.lower() or "hierarchy" in line.lower():
                    heading_match = self.alt_heading_pattern.search(line)
                    if heading_match:
                        heading_text = heading_match.group(1).strip()
                        heading_hierarchy = [h.strip() for h in heading_text.split('>')]
                    else:
                        # Try to extract hierarchy from the line itself
                        if ">" in line:
                            potential_hierarchy = [h.strip() for h in line.split('>')]
                            if len(potential_hierarchy) >= 2:
                                heading_hierarchy = potential_hierarchy
                
                i += 1
            
            # If we found a heading, look for content
            if heading_hierarchy:
                content_lines = []
                
                # Collect lines until we find something that looks like a continuation flag
                # or the next heading
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check if this looks like a continuation flag
                    if ("continue" in line.lower() or "continues" in line.lower()) and \
                       ("true" in line.lower() or "false" in line.lower() or "partial" in line.lower()):
                        continuation_match = self.alt_continuation_pattern.search(line)
                        if continuation_match:
                            continuation_flag = continuation_match.group(1)
                        else:
                            # Try to extract directly
                            if "true" in line.lower():
                                continuation_flag = "True"
                            elif "false" in line.lower():
                                continuation_flag = "False"
                            elif "partial" in line.lower():
                                continuation_flag = "Partial"
                        
                        i += 1
                        break
                    
                    # Check if this looks like the next heading
                    if ">" in line or "heading" in line.lower() or "hierarchy" in line.lower():
                        heading_match = self.alt_heading_pattern.search(line)
                        if heading_match or (i < len(lines) - 1 and "content" in lines[i+1].lower()):
                            # This is probably the next heading
                            break
                    
                    content_lines.append(line)
                    i += 1
                
                # Create a chunk with what we've found
                content = "\n".join(content_lines).strip()
                
                if content:  # Only create chunk if we have content
                    chunk = Chunk(
                        id=f"b{batch_idx}_c{len(chunks)}_{uuid.uuid4().hex[:8]}",
                        content=content,
                        heading_hierarchy=heading_hierarchy,
                        page_numbers=page_numbers,
                        continuation_flag=continuation_flag,
                        source_batch=batch_idx,
                        metadata={
                            "position_in_batch": len(chunks),
                            "raw_length": len(content),
                            "heading_count": len(heading_hierarchy),
                            "extraction_method": "alternative",
                        }
                    )
                    chunks.append(chunk)
            else:
                # If we didn't find a heading, move on
                i += 1
        
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
        
        # Create merged chunk with a shorter ID to avoid filename length issues
        return Chunk(
            id=f"m{chunk1.source_batch}_{uuid.uuid4().hex[:8]}",
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
