#!/usr/bin/env python3
import os
import argparse
import logging
import sys
from pathlib import Path

from src.orchestrator import process_pdf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("vision_chunking.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for the Vision-Guided Chunking system.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Vision-Guided Chunking: Intelligent PDF document processing"
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input PDF file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to save the output JSON file. If not provided, will use the input filename with .json extension."
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    if not input_path.lower().endswith('.pdf'):
        logger.error(f"Input file must be a PDF: {input_path}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # Use input filename with .json extension in the output directory
        input_filename = os.path.basename(input_path)
        input_name = os.path.splitext(input_filename)[0]
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{input_name}_chunks.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Starting Vision-Guided Chunking on {input_path}")
    logger.info(f"Output will be saved to {output_path}")
    
    try:
        # Process the PDF
        process_pdf(input_path, output_path)
        logger.info(f"Processing complete. Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
