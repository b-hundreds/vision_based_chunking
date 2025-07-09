import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_OUTPUT_TOKENS = 65536

# PDF processing configuration
PDF_BATCH_SIZE = 4  # Number of PDF pages to process in each batch
PDF_IMAGE_DPI = 300  # DPI for PDF to image conversion

# Paths
PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
    "multimodal_chunking_prompt.txt"
)
