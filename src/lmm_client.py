"""
Large Multimodal Model (LMM) client for processing document batches.
"""

import os
import base64
import logging
from typing import List, Dict, Any, Optional, Union
import json
from io import BytesIO
from pathlib import Path
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from .batch import PageImage
from .context import Context

logger = logging.getLogger(__name__)


class LMMClient:
    """
    Client for interacting with Large Multimodal Models (LMMs).
    
    This class handles:
    1. Preparing inputs (images and prompts) for the LMM
    2. Calling the LMM API
    3. Processing and validating responses
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        max_tokens: int = 16000,
        temperature: float = 0.2,
    ):
        """
        Initialize the LMM client.
        
        Args:
            model_name: Name of the LMM model to use
            api_key: API key for the model provider
            max_tokens: Maximum tokens in the LMM response
            temperature: Temperature for generation (lower = more deterministic)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key and "gemini" in model_name.lower():
            self.api_key = os.getenv("GOOGLE_API_KEY")
        elif not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
            
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set it in the constructor or "
                "as GOOGLE_API_KEY or OPENAI_API_KEY environment variable."
            )
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._init_client()
    
    def _init_client(self):
        """Initialize the appropriate client based on model name."""
        if "gemini" in self.model_name.lower():
            self._init_gemini_client()
        else:
            self._init_openai_client()
    
    def _init_gemini_client(self):
        """Initialize Google Gemini client."""
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            self.client = genai
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "max_output_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
            )
            logger.info(f"Initialized Gemini client with model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install it with: pip install google-generativeai"
            )
    
    def _init_openai_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=self.api_key)
            self.model = self.model_name
            logger.info(f"Initialized OpenAI client with model: {self.model_name}")
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install it with: pip install openai"
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def process_batch(
        self,
        batch_pages: List[PageImage],
        context: Context,
    ) -> str:
        """
        Process a batch of document pages using the LMM.
        
        Args:
            batch_pages: List of page images in the batch
            context: Context information from previous batches
            
        Returns:
            Raw LMM output as a string
        """
        # Prepare prompt and images
        prompt = self._create_prompt(context)
        images = [self._prepare_image(page.image) for page in batch_pages]
        
        # Call appropriate API based on model type
        batch_info = f"Batch with pages {[page.page_number for page in batch_pages]}"
        logger.info(f"Sending {len(images)} page images to LMM for {batch_info}")
        
        if "gemini" in self.model_name.lower():
            response = self._call_gemini_api(prompt, images)
        else:
            response = self._call_openai_api(prompt, images)
        
        # Log stats about the response to help with debugging
        chunk_count = response.count("[CHUNK]")
        logger.info(f"Received LMM response with {len(response)} characters and {chunk_count} chunk markers")
        
        # Check if the response seems valid
        if chunk_count == 0:
            logger.warning(f"No [CHUNK] markers found in LMM response for {batch_info}")
            logger.debug(f"Response snippet: {response[:500]}...")
        
        return response
    
    def _create_prompt(self, context: Context) -> str:
        """
        Create a prompt for the LMM based on context.
        
        Args:
            context: Context information from previous batches
            
        Returns:
            Formatted prompt string
        """
        # The detailed prompt instructions are crucial for the quality of chunking
        # This is a simplified version; the full version would be more comprehensive
        
        # First part: Task description
        task_description = """
You are an expert document analyzer that can see both text and visual layout in document pages. 
Your task is to intelligently chunk these document pages into meaningful, contextually rich segments.

Unlike traditional text-only chunking, you will use your visual understanding to:
1. Respect document layout (columns, tables, headers, footers)
2. Preserve logical content boundaries (paragraphs, sections, lists)
3. Maintain continuity across pages
4. Keep related elements together (table rows with headers, lists items)

IMPORTANT: You MUST return ALL content from the pages as properly formatted chunks, even if the content seems unimportant.
Every page needs to be processed and returned as at least one chunk. Do not skip any content.

CRUCIAL INSTRUCTION ABOUT HEADINGS: For each chunk, you MUST use the heading hierarchy that's ACTUALLY VISIBLE on the current pages. 
DO NOT use headings from previous pages unless they are the current active section headings. 
Each heading in the hierarchy should be from the current section structure visible in these pages.
"""

        # Second part: Context information
        context_section = ""
        if context.summary or context.last_chunk or context.heading_hierarchy:
            context_section = """
PREVIOUS CONTEXT:
"""
            if context.summary:
                context_section += f"Summary of previous pages: {context.summary}\n\n"
            
            if context.heading_hierarchy:
                context_section += f"Heading hierarchy from previous pages: {' > '.join(context.heading_hierarchy)}\n"
                context_section += "IMPORTANT: Only use this previous heading hierarchy if it's still relevant to the current pages. If the current pages start a new section with new headings, use those instead.\n\n"
                
            if context.last_chunk:
                context_section += f"Last chunk from previous pages: {context.last_chunk}\n\n"

        # Third part: Chunk formatting instructions
        format_instructions = """
OUTPUT FORMAT:
For each chunk, output in the following format:

[CHUNK]
[HEADING_HIERARCHY]Document Title > Section > Subsection[/HEADING_HIERARCHY]
[CONTENT]
The actual content of the chunk goes here. This should be a cohesive, meaningful unit.
[/CONTENT]
[CONTINUES]True|False|Partial[/CONTINUES]
[/CHUNK]

YOU MUST use the EXACT tags shown above, with the square brackets. Each chunk must have:
1. A [CHUNK] opening tag and [/CHUNK] closing tag
2. A [HEADING_HIERARCHY] section with proper hierarchy (even if it's just the document title)
   - This MUST reflect the actual headings visible on these specific pages
   - Do NOT carry over headings from previous pages unless they are still the active section headings
   - If you can't determine the heading, use "Unknown Section" or the document title
3. A [CONTENT] section with the actual text content
4. A [CONTINUES] flag indicating if this chunk continues from a previous one

CHUNKING RULES:
1. Create logical chunks that preserve meaning and context
2. Include heading hierarchy for each chunk based on the VISIBLE headings on the current pages
3. Mark each chunk with a continuation flag:
   - [CONTINUES]True[/CONTINUES] - This chunk continues from the previous one
   - [CONTINUES]False[/CONTINUES] - This chunk starts a new topic/section
   - [CONTINUES]Partial[/CONTINUES] - Uncertain relationship

SPECIAL HANDLING:
- Tables: Create one chunk per row, but include column headers in EACH row chunk
- Lists: Keep all items in a list together in one chunk
- Images: Include descriptive text about the image and its caption
- Headers/Footers: Exclude page numbers and repeating headers/footers
- Math/Equations: Keep equations intact with their surrounding context

IF THE PAGE HAS ONLY FIGURES/TABLES: Still create a chunk for each figure or table, with a description of what the figure/table contains.

AGAIN, MAKE SURE EVERY PAGE IS REPRESENTED IN AT LEAST ONE CHUNK, AND USE THE EXACT TAG FORMAT AS SHOWN ABOVE.
"""

        # Combine all parts
        full_prompt = task_description + context_section + format_instructions
        
        return full_prompt
    
    def _prepare_image(self, image: Union[Image.Image, bytes, str]) -> Union[str, Image.Image]:
        """
        Prepare an image for the LMM API.
        
        Args:
            image: Image as PIL Image, bytes, or file path
            
        Returns:
            Prepared image in the format required by the API
        """
        # If image is a file path, load it
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            
        # If image is bytes, convert to PIL Image
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
            
        # For Gemini, return PIL Image
        if "gemini" in self.model_name.lower():
            return image
            
        # For OpenAI, convert to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    
    def _call_gemini_api(self, prompt: str, images: List[Image.Image]) -> str:
        """
        Call the Google Gemini API.
        
        Args:
            prompt: Formatted prompt string
            images: List of prepared images
            
        Returns:
            Raw response text
        """
        try:
            # Create content parts (text + images)
            content_parts = [prompt]
            for img in images:
                content_parts.append(img)
                
            # Call the model
            response = self.model.generate_content(content_parts)
            
            if not hasattr(response, "text"):
                raise ValueError(f"Unexpected response format: {response}")
                
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            raise
    
    def _call_openai_api(self, prompt: str, images: List[str]) -> str:
        """
        Call the OpenAI API.
        
        Args:
            prompt: Formatted prompt string
            images: List of base64-encoded images
            
        Returns:
            Raw response text
        """
        try:
            # Prepare messages with images
            messages = [{"role": "user", "content": []}]
            
            # Add text prompt
            messages[0]["content"].append({
                "type": "text",
                "text": prompt
            })
            
            # Add images
            for img_b64 in images:
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
                
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise
