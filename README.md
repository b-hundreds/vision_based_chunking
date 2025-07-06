# Vision-Based Chunking for RAG Systems

This project implements an advanced document chunking algorithm for Retrieval-Augmented Generation (RAG) systems using Large Multimodal Models (LMMs). The algorithm processes PDF documents as images, allowing it to understand both the visual layout and textual content simultaneously.

## Key Features

- **Multimodal Processing**: Uses LMMs to "see" the visual layout and "read" content at the same time
- **Batch Processing**: Handles multi-page segments to preserve context across pages
- **Context Preservation**: Maintains heading hierarchies and continuity between chunks
- **Intelligent Chunking Rules**:
  - Preserves structural elements (tables, lists, diagrams)
  - Respects content flow and logical boundaries
  - Generates continuation flags to indicate relationships between chunks
- **Enhanced Metadata**: Each chunk includes structural context for better retrieval

## Architecture

The system follows these main processing steps:

1. **Batch Creation**: Split document into batches of configurable page count
2. **Context Management**: Preserve context between batches
3. **LMM Processing**: Process pages visually using a multimodal model
4. **Smart Chunking**: Create chunks following sophisticated rules
5. **Post-processing**: Validate and merge related chunks
6. **Embedding & Storage**: Convert chunks to vectors and store in a vector database

## Setup and Usage

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Chunker

```python
from vision_based_chunking import VisionChunker

# Initialize the chunker
chunker = VisionChunker(
    batch_size=4,  # Number of pages per batch
    lmm_model="gemini-2.5-pro",  # LMM model to use
    embedding_model="text-embedding-3-small",  # Embedding model
)

# Process a document
chunks = chunker.process_document("path/to/document.pdf")

# Store chunks in vector database
chunker.ingest_chunks(chunks, vector_db="elasticsearch")
```

## Project Structure

```
vision_based_chunking/
├── src/                   # Source code
│   ├── __init__.py
│   ├── chunker.py         # Main chunking logic
│   ├── batch.py           # Batch processing
│   ├── context.py         # Context management
│   ├── lmm_client.py      # LMM API client
│   ├── processors/        # Post-processing components
│   ├── storage/           # Vector database integration
│   └── utils/             # Utility functions
├── data/                  # Sample data and test documents
├── tests/                 # Test suite
├── .env                   # Environment variables (not versioned)
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## License

MIT
