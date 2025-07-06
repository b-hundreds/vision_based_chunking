"""
PDF utility functions for extracting pages and images.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import io

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts pages and images from PDF documents.
    
    This class is responsible for:
    1. Converting PDF pages to images
    2. Extracting basic metadata from PDFs
    3. Providing a list of page images for batch processing
    """
    
    def __init__(self, dpi: int = 300):
        """
        Initialize the PDF extractor.
        
        Args:
            dpi: Resolution for rendering PDF pages (default: 300)
        """
        self.dpi = dpi
    
    def extract_page_images(
        self,
        pdf_path: Union[str, Path],
    ) -> List[Any]:  # List[PageImage]
        """
        Extract page images from a PDF document.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            List of PageImage objects
        """
        try:
            from pypdf import PdfReader
            from ..batch import PageImage
        except ImportError:
            raise ImportError(
                "pypdf package not installed. "
                "Install it with: pip install pypdf"
            )
            
        try:
            # Import here to avoid circular imports
            from PIL import Image
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF package not installed. "
                "Install it with: pip install pymupdf"
            )
            
        pdf_path = Path(pdf_path)
        logger.info(f"Extracting images from PDF: {pdf_path}")
        
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        page_images = []
        
        # Extract each page as an image
        for page_number, page in enumerate(pdf_document):
            # Render page to a pixmap
            pix = page.get_pixmap(dpi=self.dpi)
            
            # Convert pixmap to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Create PageImage object
            page_image = PageImage(
                image=img,
                page_number=page_number + 1,  # 1-based page numbers
                width=img.width,
                height=img.height,
                dpi=self.dpi,
            )
            
            page_images.append(page_image)
            
        logger.info(f"Extracted {len(page_images)} page images from PDF")
        return page_images
    
    def get_pdf_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from a PDF document.
        
        Args:
            pdf_path: Path to the PDF document
            
        Returns:
            Dictionary of metadata
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf package not installed. "
                "Install it with: pip install pypdf"
            )
            
        pdf_path = Path(pdf_path)
        try:
            # Open the PDF
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                info = reader.metadata
                
                # Extract basic metadata
                metadata = {
                    "title": info.title if info.title else None,
                    "author": info.author if info.author else None,
                    "subject": info.subject if info.subject else None,
                    "creator": info.creator if info.creator else None,
                    "producer": info.producer if info.producer else None,
                    "page_count": len(reader.pages),
                }
                logger.info(f"Extracted PDF metadata: {metadata}")
                return metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {"error": str(e)}
