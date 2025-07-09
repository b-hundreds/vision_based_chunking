import re
import logging
from typing import List

from src.data_models import RawChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChunkParser:
    """
    Parses raw text from the LLM into structured chunks.
    """
    
    def __init__(self):
        """Initialize the chunk parser."""
        # Regex pattern to match chunks in the LLM output
        self.chunk_pattern = re.compile(
            r'CHUNK\s+\[continues:\s*(true|false|partial)\]\s*\n'
            r'HEADING:\s*(.*?)\s*\n'
            r'CONTENT:\s*\n(.*?)\nEND CHUNK',
            re.DOTALL
        )
    
    def parse_llm_response(self, llm_response: str) -> List[RawChunk]:
        """
        Parse the raw LLM response into a list of RawChunk objects.
        
        Args:
            llm_response: Raw text response from the LLM
            
        Returns:
            List of parsed RawChunk objects
        """
        logger.info("Parsing LLM response into chunks")
        
        # Find all matches in the response
        matches = self.chunk_pattern.findall(llm_response)
        
        if not matches:
            logger.warning("No chunks found in LLM response. Response format may be incorrect.")
            # Log a portion of the response for debugging
            logger.debug(f"Response preview: {llm_response[:200]}...")
            return []
        
        chunks = []
        for continues_flag, heading, content in matches:
            # Convert continues_flag to the right type
            if continues_flag == "true":
                continues = True
            elif continues_flag == "false":
                continues = False
            else:  # "partial"
                continues = "partial"
            
            # Create a RawChunk object
            chunk = RawChunk(
                continues=continues,
                heading=heading.strip(),
                content=content.strip()
            )
            chunks.append(chunk)
        
        logger.info(f"Parsed {len(chunks)} chunks from LLM response")
        return chunks
