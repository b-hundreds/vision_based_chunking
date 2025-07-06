"""
Simple chunking script that only performs document chunking without embedding or database storage.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import the simple chunker
from src import get_simple_vision_chunker
SimpleVisionChunker = get_simple_vision_chunker()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def main():
    """Run the simple chunking process."""
    
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        logger.error(
            "API keys not found. Please set GOOGLE_API_KEY or OPENAI_API_KEY "
            "in your .env file or environment variables."
        )
        return
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Vision-Based Chunking")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document")
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output directory for chunks (default: output)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of pages per batch (default: 4)"
    )
    parser.add_argument(
        "--lmm-model", type=str, default="gemini-2.5-pro",
        help="LMM model to use (default: gemini-2.5-pro)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the simple chunker
    chunker = SimpleVisionChunker(
        batch_size=args.batch_size,
        lmm_model=args.lmm_model,
    )
    
    # Process the document
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
        
    logger.info(f"Processing document: {pdf_path}")
    chunks = chunker.process_document(
        document_path=pdf_path,
        output_dir=output_dir,
    )
    
    # Print some statistics
    logger.info(f"Created {len(chunks)} chunks")
    
    # Print sample chunks
    if chunks:
        logger.info("Sample chunk:")
        sample_chunk = chunks[0]
        logger.info(f"ID: {sample_chunk.id}")
        logger.info(f"Heading: {' > '.join(sample_chunk.heading_hierarchy)}")
        logger.info(f"Page numbers: {sample_chunk.page_numbers}")
        logger.info(f"Content sample: {sample_chunk.content[:100]}...")
    
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
