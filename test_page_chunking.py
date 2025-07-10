#!/usr/bin/env python3
import os
import sys
import logging
from PIL import Image

from src.llm_handler import LLMHandler
from src.chunk_parser import ChunkParser
from src.post_processor import PostProcessor
from src.pdf_processor import PDFProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("test_chunking.log")
    ]
)
logger = logging.getLogger(__name__)

def test_single_page(pdf_path, page_number=1):
    """
    Test the chunking system on a single page of a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number to test (1-indexed)
    """
    logger.info(f"Testing chunking on page {page_number} of {pdf_path}")
    
    # Initialize components
    pdf_processor = PDFProcessor(pdf_path)
    llm_handler = LLMHandler()
    chunk_parser = ChunkParser()
    post_processor = PostProcessor()
    
    # Load the PDF
    pdf_processor.load_pdf()
    
    # Get the specified page
    if page_number > len(pdf_processor.pages):
        logger.error(f"Page number {page_number} exceeds the number of pages in the PDF ({len(pdf_processor.pages)})")
        return
    
    page_image = pdf_processor.pages[page_number - 1]
    
    # Process the page with LLM
    llm_response = llm_handler.process_batch(
        images=[page_image],
        batch_index=0,
        total_batches=1
    )
    
    # Parse LLM response
    raw_chunks = chunk_parser.parse_llm_response(llm_response)
    
    # Process chunks
    final_chunks = post_processor.process_chunks(raw_chunks, [page_number])
    
    # Print results
    logger.info(f"Generated {len(final_chunks)} chunks from page {page_number}")
    
    for i, chunk in enumerate(final_chunks):
        logger.info(f"Chunk {i+1}:")
        logger.info(f"  Heading: {chunk.heading}")
        logger.info(f"  Content (preview): {chunk.content[:100]}...")
        logger.info("-" * 50)
    
    return final_chunks

if __name__ == "__main__":
    # Path to your test PDF
    pdf_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "input_pdfs",
        "test_paper.pdf"
    )
    
    # Test page 2 (usually contains abstract and introduction)
    test_single_page(pdf_path, page_number=2)
