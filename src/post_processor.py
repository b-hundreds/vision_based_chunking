import logging
import uuid
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
        pass
    
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
            
            # Handle continuation chunks
            if chunk.continues and last_chunk is not None:
                # Merge this chunk with the last one
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
                    heading=chunk.heading,
                    content=chunk.content,
                    main_heading=main_heading,
                    section_heading=section_heading,
                    chunk_title=chunk_title,
                    page_numbers=[page_numbers[min(i, len(page_numbers) - 1)]]
                )
                final_chunks.append(new_chunk)
                last_chunk = new_chunk
                logger.debug(f"Created new chunk: {new_chunk.id}")
        
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
        
        last_chunk = final_chunks[-1]
        
        # Create a context string with heading hierarchy and the last part of content
        context = (
            f"Previous chunk information:\n"
            f"Heading: {last_chunk.heading}\n"
            f"Last content: {last_chunk.content[-500:] if len(last_chunk.content) > 500 else last_chunk.content}\n"
        )
        
        return context
