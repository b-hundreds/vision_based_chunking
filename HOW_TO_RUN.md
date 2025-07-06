# How to Run the Example

This guide explains how to run the example script to test the vision-based chunking algorithm.

## Prerequisites

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root directory with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   You need at least one of these keys:
   - `GOOGLE_API_KEY`: Required if using Gemini models
   - `OPENAI_API_KEY`: Required if using OpenAI models or embeddings

3. Optional: For vector database storage, add these to your `.env` file:
   ```
   VECTOR_DB_TYPE=elasticsearch
   VECTOR_DB_HOST=localhost
   VECTOR_DB_PORT=9200
   VECTOR_DB_USERNAME=
   VECTOR_DB_PASSWORD=
   VECTOR_DB_INDEX=vision_chunks
   ```

## Running the Example

The example script processes a PDF document using the vision-based chunking algorithm.

### Basic Usage

```bash
python example.py path/to/your/document.pdf
```

This will process the document and save the chunks to the `output` directory.

### Advanced Options

The script supports several command-line options:

```bash
python example.py path/to/your/document.pdf --output custom_output --batch-size 2 --lmm-model gemini-2.5-pro --embedding-model text-embedding-3-small --ingest
```

Options:
- `--output`: Directory to save the chunks (default: `output`)
- `--batch-size`: Number of pages per batch (default: 4)
- `--lmm-model`: LMM model to use (default: `gemini-2.5-pro`)
- `--embedding-model`: Embedding model to use (default: `text-embedding-3-small`)
- `--ingest`: Flag to ingest chunks into the vector database (requires vector DB configuration)

## Output

The script creates the following output:
- Individual JSON files for each chunk in the output directory
- A summary file (`chunks_summary.json`) with an overview of all chunks

## Troubleshooting

1. **ImportError: No module named 'xyz'**
   - Make sure you have installed all dependencies with `pip install -r requirements.txt`

2. **API Key Errors**
   - Check that your `.env` file contains the correct API keys

3. **PDF Processing Errors**
   - Ensure you have a valid PDF file
   - Try with a simpler PDF if you encounter issues with complex documents

4. **Vector Database Errors**
   - Make sure Elasticsearch is running if you're using the `--ingest` option
   - Check your vector database configuration in the `.env` file
