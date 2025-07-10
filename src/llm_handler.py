import os
import logging
from typing import List, Optional, Dict
from PIL import Image

import google.generativeai as genai
from src.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS, PROMPT_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini API
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
genai.configure(api_key=GEMINI_API_KEY)


class LLMHandler:
    """
    Handles communication with the Gemini API.
    """
    
    def __init__(self):
        """
        Initialize the LLM handler.
        """
        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config={
                "temperature": GEMINI_TEMPERATURE,
                "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
            }
        )
        
        # Load prompt template
        with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
    def format_prompt(self, context: Optional[str] = None, batch_info: Optional[Dict] = None) -> str:
        """
        Format the prompt template with the provided context.
        
        Args:
            context: Optional context information to include in the prompt
            batch_info: Optional metadata about the current batch
            
        Returns:
            Formatted prompt string
        """
        # Start with the basic context
        context_section = context if context else "No previous context available."
        
        # Add batch metadata information if available
        if batch_info:
            batch_metadata = (
                f"\nBatch Information:\n"
                f"- Page numbers in this batch: {batch_info.get('page_numbers', 'Unknown')}\n"
                f"- This is batch #{batch_info.get('batch_index', 'Unknown')} of {batch_info.get('total_batches', 'Unknown')}\n"
            )
            
            # Add specific guidance for maintaining heading continuity
            continuity_guidance = (
                f"\nCONTINUITY GUIDANCE:\n"
                f"- Maintain consistent section headings across page boundaries\n"
                f"- For figures, tables, or charts that span multiple pages, use the same section heading\n"
                f"- Pay special attention to content that might be split across page boundaries and maintain proper heading context\n"
            )
            
            context_section += batch_metadata + continuity_guidance
        
        return self.prompt_template.format(context=context_section)
    
    def process_batch(self, images: List[Image.Image], context: Optional[str] = None, batch_index: int = 0, total_batches: int = 0) -> str:
        """
        Process a batch of images with the LLM.
        
        Args:
            images: List of PIL Image objects to process
            context: Optional context from previous batch
            batch_index: Current batch index (0-based)
            total_batches: Total number of batches
            
        Returns:
            Raw text response from the LLM
        """
        # Create batch metadata to provide additional context
        batch_info = {
            "page_numbers": [i+1 for i in range(len(images))],
            "batch_index": batch_index + 1,
            "total_batches": total_batches
        }
        
        formatted_prompt = self.format_prompt(context, batch_info)
        logger.info(f"Sending batch of {len(images)} images to Gemini (batch {batch_index+1}/{total_batches})")
        
        try:
            response = self.model.generate_content([formatted_prompt] + images)
            logger.info("Successfully received response from Gemini")
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
