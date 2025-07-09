from typing import List, Tuple, Dict
import os
from pdf2image import convert_from_path
from PIL import Image
import logging

from src.config import PDF_BATCH_SIZE, PDF_IMAGE_DPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Responsible for processing PDF files into batches of images.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = pdf_path
        self.filename = os.path.basename(pdf_path).split('.')[0]
        self.pages = None
        
    def load_pdf(self) -> None:
        """
        Load the PDF file and convert all pages to images.
        """
        logger.info(f"Loading PDF from {self.pdf_path}")
        try:
            self.pages = convert_from_path(
                self.pdf_path, 
                dpi=PDF_IMAGE_DPI
            )
            logger.info(f"Successfully loaded {len(self.pages)} pages from PDF")
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
            
    def get_batches(self) -> List[Tuple[List[Image.Image], List[int]]]:
        """
        Divide the PDF pages into batches.
        
        Returns:
            A list of tuples, each containing:
            - A list of page images in the batch
            - A list of page numbers (1-indexed) in the batch
        """
        if self.pages is None:
            self.load_pdf()
            
        batches = []
        for i in range(0, len(self.pages), PDF_BATCH_SIZE):
            # Get a batch of pages
            batch_pages = self.pages[i:i+PDF_BATCH_SIZE]
            # Get page numbers (1-indexed)
            batch_page_numbers = list(range(i+1, i+1+len(batch_pages)))
            
            batches.append((batch_pages, batch_page_numbers))
            
        logger.info(f"PDF divided into {len(batches)} batches")
        return batches
