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
        
    def format_prompt(self, context: Optional[str] = None) -> str:
        """
        Format the prompt template with the provided context.
        
        Args:
            context: Optional context information to include in the prompt
            
        Returns:
            Formatted prompt string
        """
        if context:
            return self.prompt_template.format(context=context)
        else:
            return self.prompt_template.format(context="No previous context available.")
    
    def process_batch(self, images: List[Image.Image], context: Optional[str] = None) -> str:
        """
        Process a batch of images with the LLM.
        
        Args:
            images: List of PIL Image objects to process
            context: Optional context from previous batch
            
        Returns:
            Raw text response from the LLM
        """
        formatted_prompt = self.format_prompt(context)
        logger.info(f"Sending batch of {len(images)} images to Gemini")
        
        try:
            response = self.model.generate_content([formatted_prompt] + images)
            logger.info("Successfully received response from Gemini")
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
