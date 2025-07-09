# Vision-Guided Chunking

A system for intelligent document chunking using visual information from PDFs.

## Overview

Vision-Guided Chunking overcomes limitations of traditional text-based chunking methods by "looking" at PDF documents, understanding their visual structure (layout, tables, images, columns), and then dividing them into meaningful, coherent chunks of information.

## Features

- **Visual Understanding**: Processes PDF documents as images, preserving layout information
- **Intelligent Chunking**: Creates semantically meaningful chunks based on document structure
- **Context Preservation**: Maintains context across page boundaries and between processing batches
- **Hierarchical Headings**: Provides a 3-level heading structure for each chunk
- **Multimodal Processing**: Uses Google Gemini to analyze both visual and textual information

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/vision_guided_chunking.git
cd vision_guided_chunking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Google Gemini API key:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Usage

Process a PDF file:

```bash
python main.py -i input_pdfs/your_document.pdf -o output/your_document_chunks.json
```

Or simply:

```bash
python main.py -i input_pdfs/your_document.pdf
```

This will automatically save the output to `output/your_document_chunks.json`.

## System Architecture

The system follows this workflow:

1. **Input**: A PDF file
2. **Preparation**: Convert PDF pages to images to preserve layout
3. **Batching**: Group images into small batches (e.g., 4 pages per batch)
4. **LLM Processing**: Send each batch to Google Gemini with a detailed prompt
5. **Context Preservation**: Include context from previous batches
6. **Post-processing**: Parse and merge chunks across batches
7. **Output**: A JSON file with structured, meaningful chunks

## Project Structure

```
vision_guided_chunking/
│
├── input_pdfs/                 # Input PDF files
│
├── output/                     # Output JSON files
│
├── prompts/                    # LLM prompts
│   └── multimodal_chunking_prompt.txt
│
├── src/                        # Source code
│   ├── config.py               # Configuration settings
│   ├── data_models.py          # Data structure definitions
│   ├── pdf_processor.py        # PDF to image conversion
│   ├── llm_handler.py          # Gemini API integration
│   ├── chunk_parser.py         # Parse LLM responses
│   ├── post_processor.py       # Process and merge chunks
│   └── orchestrator.py         # Main processing workflow
│
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
└── .env                        # Environment variables (API keys)
```

## License

MIT
