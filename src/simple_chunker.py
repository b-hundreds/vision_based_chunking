"""
Simplified chunking implementation that only performs document chunking 
without embedding or vector database storage.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
import uuid
from pathlib import Path
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

from .batch import BatchProcessor
from .context import ContextManager
from .lmm_client import LMMClient
from .processors import ChunkProcessor
from .utils.pdf import PDFExtractor
from .models import Chunk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleVisionChunker:
    """
    Simplified chunker class that only implements the vision-based chunking algorithm 
    without embedding or database storage.
    
    This class orchestrates the chunking process:
    1. Splits documents into batches
    2. Manages context between batches
    3. Processes batches using LMM
    4. Post-processes chunks
    5. Saves chunks as JSON files
    """
    
    def __init__(
        self,
        batch_size: int = None,
        lmm_model: str = None,
    ):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            batch_size: Number of pages per batch
            lmm_model: Name of the LMM model to use
        """
        # Load parameters from environment if not provided
        self.batch_size = batch_size or int(os.getenv("BATCH_SIZE", "4"))
        self.lmm_model = lmm_model or os.getenv("LMM_MODEL", "gemini-2.5-pro")
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.batch_processor = BatchProcessor(batch_size=self.batch_size)
        self.context_manager = ContextManager()
        self.lmm_client = LMMClient(model_name=self.lmm_model)
        self.chunk_processor = ChunkProcessor()
    
    def process_document(
        self, 
        document_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        return_chunks: bool = True,
    ) -> List[Chunk]:
        """
        Process a document using the vision-based chunking algorithm.
        
        Args:
            document_path: Path to the PDF document
            output_dir: Directory to save processed chunks (optional)
            return_chunks: Whether to return the chunks as a list
            
        Returns:
            List of processed chunks (if return_chunks is True)
        """
        logger.info(f"Processing document: {document_path}")
        
        # Extract pages as images
        document_path = Path(document_path)
        page_images = self.pdf_extractor.extract_page_images(document_path)
        
        # Create batches of pages
        batches = self.batch_processor.create_batches(page_images)
        logger.info(f"Created {len(batches)} batches of pages")
        
        # Process each batch with context preservation
        all_chunks = []
        previous_context = None
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx+1}/{len(batches)}")
            
            # Get context from previous batch
            if previous_context:
                context = previous_context
            else:
                context = self.context_manager.create_initial_context()
                
            # Process batch with LMM
            raw_lmm_output = self.lmm_client.process_batch(
                batch_pages=batch,
                context=context,
            )
            
            # Extract and validate chunks
            batch_chunks = self.chunk_processor.process(
                raw_lmm_output=raw_lmm_output,
                batch_idx=batch_idx,
                page_numbers=[page.page_number for page in batch],
            )
            
            # If no chunks were extracted, try again with a different approach
            retry_count = 0
            while len(batch_chunks) == 0 and retry_count < 1:
                logger.warning(f"No chunks extracted from batch {batch_idx}, retrying with simplified prompt")
                retry_count += 1
                
                # Create a simplified context without previous context to avoid confusion
                retry_context = self.context_manager.create_initial_context()
                
                # Try again with the LMM
                raw_lmm_output = self.lmm_client.process_batch(
                    batch_pages=batch,
                    context=retry_context,
                )
                
                # Try to extract chunks again
                batch_chunks = self.chunk_processor.process(
                    raw_lmm_output=raw_lmm_output,
                    batch_idx=batch_idx,
                    page_numbers=[page.page_number for page in batch],
                )
            
            # Create a simple fallback chunk if still no chunks
            if len(batch_chunks) == 0:
                logger.warning(f"Still no chunks from batch {batch_idx} after retry, creating fallback chunk")
                
                # Create a simple fallback chunk with page numbers
                page_nums = [page.page_number for page in batch]
                fallback_content = f"Content from pages {', '.join(map(str, page_nums))}. " + \
                                   "This is a fallback chunk created because no structured content could be extracted."
                
                fallback_chunk = Chunk(
                    id=f"fallback_b{batch_idx}_{uuid.uuid4().hex[:8]}",
                    content=fallback_content,
                    heading_hierarchy=["Fallback Content"],
                    page_numbers=page_nums,
                    continuation_flag="False",
                    source_batch=batch_idx,
                    metadata={"is_fallback": True},
                )
                
                batch_chunks = [fallback_chunk]
            
            # Update context for next batch
            previous_context = self.context_manager.update_context(
                current_chunks=batch_chunks,
                previous_context=context,
            )
            
            # Collect chunks
            all_chunks.extend(batch_chunks)
            
        # Post-process all chunks (merge related chunks based on continuation flags)
        final_chunks = self.chunk_processor.post_process_all(all_chunks)
        logger.info(f"Created {len(final_chunks)} final chunks")
        
        # Save chunks if output directory is specified
        if output_dir:
            self._save_chunks(final_chunks, Path(output_dir))
        
        return final_chunks if return_chunks else None
    
    def _save_chunks(self, chunks: List[Chunk], output_dir: Path) -> None:
        """Save chunks to disk as JSON files."""
        import json
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save individual chunks
        for chunk in chunks:
            chunk_file = output_dir / f"chunk_{chunk.id}.json"
            with open(chunk_file, 'w') as f:
                # Convert dataclass to dict
                chunk_dict = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "page_numbers": chunk.page_numbers,
                    "continuation_flag": chunk.continuation_flag,
                    "source_batch": chunk.source_batch,
                    "metadata": chunk.metadata,
                }
                json.dump(chunk_dict, f, indent=2)
        
        # Save summary file
        summary_file = output_dir / "chunks_summary.json"
        with open(summary_file, 'w') as f:
            summary = {
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "id": chunk.id,
                        "heading": " > ".join(chunk.heading_hierarchy),
                        "pages": chunk.page_numbers,
                        "continuation": chunk.continuation_flag,
                    }
                    for chunk in chunks
                ]
            }
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
