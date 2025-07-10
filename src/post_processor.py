import logging
import uuid
import re
from typing import List, Dict, Tuple

from src.data_models import RawChunk, FinalChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Handles post-processing of chunks, including merging continuing chunks.
    """
    
    def __init__(self):
        """Initialize the post processor."""
        # Pattern to identify footnote sections
        self.footnote_pattern = re.compile(r'^\d+\.\s', re.MULTILINE)
        self.is_footnote_section = lambda content: bool(self.footnote_pattern.search(content))
        
        # Track heading context across batches and pages
        self.last_main_heading = None
        self.last_section_heading = None
        self.last_chunk_title = None
    
    def process_chunks(self, raw_chunks: List[RawChunk], page_numbers: List[int]) -> List[FinalChunk]:
        """
        Process a list of raw chunks into final chunks, merging continuing chunks.
        
        Args:
            raw_chunks: List of RawChunk objects to process
            page_numbers: List of page numbers corresponding to the batch
            
        Returns:
            List of processed FinalChunk objects
        """
        logger.info(f"Post-processing {len(raw_chunks)} chunks")
        
        final_chunks = []
        last_chunk = None
        
        for i, chunk in enumerate(raw_chunks):
            # Extract heading components
            heading_parts = chunk.heading.split(" > ")
            
            if len(heading_parts) != 3:
                logger.warning(f"Chunk heading '{chunk.heading}' does not have 3 parts. Using defaults.")
                # Use defaults if the heading doesn't have 3 parts
                main_heading = heading_parts[0] if heading_parts else "Unknown Main Heading"
                section_heading = heading_parts[1] if len(heading_parts) > 1 else "Unknown Section"
                chunk_title = heading_parts[2] if len(heading_parts) > 2 else "Unknown Chunk"
            else:
                main_heading = heading_parts[0]
                section_heading = heading_parts[1]
                chunk_title = heading_parts[2]
            
            # Check if this is a footnote section
            is_footnote = self.is_footnote_section(chunk.content)
            
            # For footnotes, we always create a new chunk with appropriate heading
            if is_footnote and not chunk_title.lower().startswith("footnote"):
                chunk_title = "Footnotes" if chunk_title else "Footnotes"
            
            # Apply heading continuity logic for chunks that span across pages
            if chunk.continues and self.last_section_heading is not None and not is_footnote:
                # When a chunk continues from a previous batch/page, preserve the section heading context
                if i == 0:  # First chunk in the batch
                    # Use the last section heading from the previous batch to ensure continuity
                    section_heading = self.last_section_heading
            
            # For figures or content that might be split across pages:
            # Check for visual content markers (like "Figure", "Table", etc.) in the chunk title or content
            content_lower = chunk.content.lower()
            title_lower = chunk_title.lower()
            is_visual_content = any(marker in content_lower[:100] or marker in title_lower 
                                   for marker in ["figure", "fig.", "table", "chart", "graph", "diagram", "illustration"])
                                   
            # If this is a continuation but doesn't have a clear section heading, use the previous one
            if (chunk.continues or is_visual_content) and i > 0 and main_heading == self.last_main_heading:
                # Keep the section heading consistent with the previous chunk
                section_heading = self.last_section_heading
            
            # Handle continuation chunks
            if chunk.continues and last_chunk is not None and not is_footnote:
                # Merge this chunk with the last one only if it's not a footnote
                last_chunk.content += f"\n{chunk.content}"
                
                # Add page numbers
                # Estimate which page this chunk is from based on its position in the batch
                chunk_page = page_numbers[min(i, len(page_numbers) - 1)]
                if chunk_page not in last_chunk.page_numbers:
                    last_chunk.page_numbers.append(chunk_page)
                
                logger.debug(f"Merged continuing chunk into previous chunk: {last_chunk.id}")
            else:
                # Create a new final chunk
                new_chunk = FinalChunk(
                    id=str(uuid.uuid4()),
                    heading=f"{main_heading} > {section_heading} > {chunk_title}",  # Regenerate heading with possibly modified section_heading
                    content=chunk.content,
                    main_heading=main_heading,
                    section_heading=section_heading,
                    chunk_title=chunk_title,
                    page_numbers=[page_numbers[min(i, len(page_numbers) - 1)]]
                )
                final_chunks.append(new_chunk)
                last_chunk = new_chunk
                logger.debug(f"Created new chunk: {new_chunk.id}")
            
            # Update the heading context
            self.last_main_heading = main_heading
            self.last_section_heading = section_heading
            self.last_chunk_title = chunk_title
        
        logger.info(f"Post-processing complete. Produced {len(final_chunks)} final chunks.")
        return final_chunks
    
    def extract_last_chunk_context(self, final_chunks: List[FinalChunk]) -> str:
        """
        Extract context information from the last chunk to provide to the next batch.
        
        Args:
            final_chunks: List of FinalChunk objects
            
        Returns:
            Context string for the next batch
        """
        if not final_chunks:
            return ""
        
        # Find the last non-footnote chunk to use as context
        last_main_chunk = None
        for chunk in reversed(final_chunks):
            if not self.is_footnote_section(chunk.content):
                last_main_chunk = chunk
                break
        
        # If all chunks are footnotes, use the last one
        if last_main_chunk is None and final_chunks:
            last_main_chunk = final_chunks[-1]
        
        # Create a context string with heading hierarchy and the last part of content
        context = (
            f"Previous chunk information:\n"
            f"Main Heading: {last_main_chunk.main_heading}\n"
            f"Section Heading: {last_main_chunk.section_heading}\n"
            f"Chunk Title: {last_main_chunk.chunk_title}\n"
            f"Page Numbers: {last_main_chunk.page_numbers}\n"
            f"Last content: {last_main_chunk.content[-500:] if len(last_main_chunk.content) > 500 else last_main_chunk.content}\n"
        )
        
        return context
