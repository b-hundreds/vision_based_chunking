"""
Core chunking implementation for vision-based document processing.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import os
from pathlib import Path
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

from .batch import BatchProcessor
from .context import ContextManager
from .lmm_client import LMMClient
from .processors import ChunkProcessor
from .storage import VectorDB
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


class VisionChunker:
    """
    Main chunker class that implements the vision-based chunking algorithm.
    
    This class orchestrates the entire chunking process:
    1. Splits documents into batches
    2. Manages context between batches
    3. Processes batches using LMM
    4. Post-processes chunks
    5. Embeds and stores chunks
    """
    
    def __init__(
        self,
        batch_size: int = None,
        lmm_model: str = None,
        embedding_model: str = None,
        vector_db_type: str = None,
        vector_db_config: Dict[str, Any] = None,
    ):
        """
        Initialize the chunker with configuration parameters.
        
        Args:
            batch_size: Number of pages per batch
            lmm_model: Name of the LMM model to use
            embedding_model: Name of the embedding model
            vector_db_type: Type of vector database (e.g., 'elasticsearch')
            vector_db_config: Configuration for the vector database
        """
        # Load parameters from environment if not provided
        self.batch_size = batch_size or int(os.getenv("BATCH_SIZE", "4"))
        self.lmm_model = lmm_model or os.getenv("LMM_MODEL", "gemini-2.5-pro")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # Initialize components
        self.pdf_extractor = PDFExtractor()
        self.batch_processor = BatchProcessor(batch_size=self.batch_size)
        self.context_manager = ContextManager()
        self.lmm_client = LMMClient(model_name=self.lmm_model)
        self.chunk_processor = ChunkProcessor()
        
        # Initialize vector database if specified
        self.vector_db = None
        if vector_db_type or os.getenv("VECTOR_DB_TYPE"):
            db_type = vector_db_type or os.getenv("VECTOR_DB_TYPE")
            db_config = vector_db_config or {
                "host": os.getenv("VECTOR_DB_HOST", "localhost"),
                "port": int(os.getenv("VECTOR_DB_PORT", "9200")),
                "username": os.getenv("VECTOR_DB_USERNAME", ""),
                "password": os.getenv("VECTOR_DB_PASSWORD", ""),
                "index": os.getenv("VECTOR_DB_INDEX", "vision_chunks"),
            }
            self.vector_db = VectorDB.create(db_type, db_config)
    
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
    
    def ingest_chunks(
        self,
        chunks: List[Chunk],
        vector_db: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> bool:
        """
        Embed and ingest chunks into a vector database.
        
        Args:
            chunks: List of chunks to ingest
            vector_db: Override the vector database type
            index_name: Override the index name
            
        Returns:
            Success status
        """
        if vector_db and not self.vector_db:
            # Initialize vector DB if specified here but not in constructor
            db_config = {
                "host": os.getenv("VECTOR_DB_HOST", "localhost"),
                "port": int(os.getenv("VECTOR_DB_PORT", "9200")),
                "username": os.getenv("VECTOR_DB_USERNAME", ""),
                "password": os.getenv("VECTOR_DB_PASSWORD", ""),
                "index": index_name or os.getenv("VECTOR_DB_INDEX", "vision_chunks"),
            }
            self.vector_db = VectorDB.create(vector_db, db_config)
        
        if not self.vector_db:
            logger.error("No vector database configured for ingestion")
            return False
            
        return self.vector_db.ingest_chunks(chunks, self.embedding_model)
    
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
