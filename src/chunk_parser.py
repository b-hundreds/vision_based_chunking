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
        # Regex pattern to match chunks in the new LLM output format
        # Format: [CONTINUES]True|False|Partial[/CONTINUES][HEAD]main_heading > section_heading > chunk_heading[/HEAD]chunk_content
        self.chunk_pattern = re.compile(
            r'\[CONTINUES\](True|False|Partial)\[/CONTINUES\]\s*\[HEAD\](.*?)\[/HEAD\]\s*(.*?)(?=\s*\[CONTINUES\]|\Z)',
            re.DOTALL | re.MULTILINE
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
        
        # Log the pattern being used for debugging
        logger.debug(f"Using regex pattern: {self.chunk_pattern.pattern}")
        
        # Find all matches in the response
        matches = self.chunk_pattern.findall(llm_response)
        
        logger.debug(f"Found {len(matches)} regex matches")
        
        if not matches:
            logger.warning("No chunks found in LLM response. Response format may be incorrect.")
            # Log the full response for debugging when no chunks found
            logger.warning(f"Full LLM response for debugging: {llm_response}")
            
            # Try to find CONTINUES tags to see if the format is close
            continues_tags = re.findall(r'\[CONTINUES\](.*?)\[/CONTINUES\]', llm_response)
            head_tags = re.findall(r'\[HEAD\](.*?)\[/HEAD\]', llm_response)
            logger.debug(f"Found {len(continues_tags)} CONTINUES tags and {len(head_tags)} HEAD tags")
            
            return []
        
        chunks = []
        for continues_flag, heading, content in matches:
            # Convert continues_flag to the right type
            if continues_flag.lower() == "true":
                continues = True
            elif continues_flag.lower() == "false":
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
