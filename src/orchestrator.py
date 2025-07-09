import os
import json
import logging
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm

from src.pdf_processor import PDFProcessor
from src.llm_handler import LLMHandler
from src.chunk_parser import ChunkParser
from src.post_processor import PostProcessor
from src.data_models import FinalChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str, output_path: str) -> str:
    """
    Main orchestration function to process a PDF into meaningful chunks.
    
    Args:
        pdf_path: Path to the input PDF file
        output_path: Path to save the output JSON file
        
    Returns:
        Path to the generated output file
    """
    logger.info(f"Starting processing of PDF: {pdf_path}")
    
    # Initialize components
    pdf_processor = PDFProcessor(pdf_path)
    llm_handler = LLMHandler()
    chunk_parser = ChunkParser()
    post_processor = PostProcessor()
    
    # Get batches of PDF pages
    batches = pdf_processor.get_batches()
    
    # Process each batch
    all_final_chunks = []
    last_chunk_context = None
    
    for batch_index, (batch_images, batch_page_numbers) in enumerate(tqdm(batches, desc="Processing PDF batches")):
        logger.info(f"Processing batch {batch_index+1}/{len(batches)}")
        
        # Process batch with LLM
        llm_response = llm_handler.process_batch(
            images=batch_images,
            context=last_chunk_context
        )
        
        # Parse LLM response
        raw_chunks = chunk_parser.parse_llm_response(llm_response)
        
        # Process chunks
        batch_final_chunks = post_processor.process_chunks(raw_chunks, batch_page_numbers)
        
        # Handle cross-batch continuations
        if batch_final_chunks and raw_chunks and all_final_chunks:
            # If the first chunk of this batch continues from the last batch
            if raw_chunks[0].continues:
                logger.info("First chunk in batch continues from previous batch. Merging.")
                # Get the last chunk from previous batches
                last_final_chunk = all_final_chunks[-1]
                # Merge content
                last_final_chunk.content += f"\n{batch_final_chunks[0].content}"
                # Add page numbers
                for page in batch_final_chunks[0].page_numbers:
                    if page not in last_final_chunk.page_numbers:
                        last_final_chunk.page_numbers.append(page)
                # Remove the first chunk as it's now merged
                batch_final_chunks = batch_final_chunks[1:]
        
        # Add the processed chunks to our collection
        all_final_chunks.extend(batch_final_chunks)
        
        # Extract context for the next batch
        if batch_final_chunks:
            last_chunk_context = post_processor.extract_last_chunk_context(batch_final_chunks)
    
    # Sort chunks by page numbers for a consistent order
    all_final_chunks.sort(key=lambda c: c.page_numbers[0] if c.page_numbers else 999999)
    
    # Save results to JSON
    result_dict = {
        "document_name": os.path.basename(pdf_path),
        "total_pages": sum(len(batch[0]) for batch in batches),
        "chunks": [chunk.to_dict() for chunk in all_final_chunks]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing complete. Generated {len(all_final_chunks)} chunks.")
    logger.info(f"Output saved to {output_path}")
    
    return output_path
