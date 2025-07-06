"""
Test suite for the vision-based chunker.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.chunker import VisionChunker, Chunk
from src.batch import BatchProcessor, PageImage
from src.context import ContextManager, Context
from src.processors.chunk_processor import ChunkProcessor


class TestBatchProcessor(unittest.TestCase):
    """Tests for the BatchProcessor class."""
    
    def test_create_batches(self):
        """Test creating batches from a list of pages."""
        # Create test pages
        pages = [
            PageImage(image=None, page_number=i+1, width=100, height=100)
            for i in range(10)
        ]
        
        # Create batches with different batch sizes
        processor1 = BatchProcessor(batch_size=4)
        batches1 = processor1.create_batches(pages)
        
        processor2 = BatchProcessor(batch_size=2)
        batches2 = processor2.create_batches(pages)
        
        # Check batch counts
        self.assertEqual(len(batches1), 3)  # 10/4 = 2 full batches + 1 partial
        self.assertEqual(len(batches2), 5)  # 10/2 = 5 full batches
        
        # Check batch sizes
        self.assertEqual(len(batches1[0]), 4)
        self.assertEqual(len(batches1[1]), 4)
        self.assertEqual(len(batches1[2]), 2)
        
        # Check page order
        self.assertEqual(batches1[0][0].page_number, 1)
        self.assertEqual(batches1[1][0].page_number, 5)
        self.assertEqual(batches1[2][0].page_number, 9)


class TestContextManager(unittest.TestCase):
    """Tests for the ContextManager class."""
    
    def test_create_initial_context(self):
        """Test creating initial context."""
        manager = ContextManager()
        context = manager.create_initial_context()
        
        self.assertIsInstance(context, Context)
        self.assertEqual(context.summary, "")
        self.assertEqual(context.last_chunk, "")
        self.assertEqual(context.heading_hierarchy, [])
        self.assertEqual(context.metadata, {})
    
    def test_update_context(self):
        """Test updating context based on chunks."""
        # Create test chunks
        chunks = [
            Chunk(
                id="chunk1",
                content="Content 1",
                heading_hierarchy=["Doc", "Section 1"],
                page_numbers=[1],
                continuation_flag="False",
                source_batch=0,
                metadata={}
            ),
            Chunk(
                id="chunk2",
                content="Content 2",
                heading_hierarchy=["Doc", "Section 2"],
                page_numbers=[2],
                continuation_flag="False",
                source_batch=0,
                metadata={}
            )
        ]
        
        # Create and update context
        manager = ContextManager()
        initial_context = manager.create_initial_context()
        updated_context = manager.update_context(chunks, initial_context)
        
        # Check updated context
        self.assertIsInstance(updated_context, Context)
        self.assertNotEqual(updated_context.summary, "")
        self.assertEqual(updated_context.last_chunk, "Content 2")
        self.assertEqual(updated_context.heading_hierarchy, ["Doc", "Section 2"])
        self.assertIn("last_page_processed", updated_context.metadata)
        self.assertIn("document_structure", updated_context.metadata)


class TestChunkProcessor(unittest.TestCase):
    """Tests for the ChunkProcessor class."""
    
    def test_process_raw_output(self):
        """Test processing raw LMM output."""
        # Create a sample LMM output
        raw_output = """
[CHUNK]
[HEADING_HIERARCHY]Document > Chapter 1 > Section 1.1[/HEADING_HIERARCHY]
[CONTENT]
This is the content of the first chunk.
[/CONTENT]
[CONTINUES]False[/CONTINUES]
[/CHUNK]

[CHUNK]
[HEADING_HIERARCHY]Document > Chapter 1 > Section 1.2[/HEADING_HIERARCHY]
[CONTENT]
This is the content of the second chunk.
[/CONTENT]
[CONTINUES]True[/CONTINUES]
[/CHUNK]
"""
        
        processor = ChunkProcessor()
        chunks = processor.process(raw_output, batch_idx=0, page_numbers=[1, 2])
        
        # Check the processed chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].heading_hierarchy, ["Document", "Chapter 1", "Section 1.1"])
        self.assertEqual(chunks[0].content, "This is the content of the first chunk.")
        self.assertEqual(chunks[0].continuation_flag, "False")
        self.assertEqual(chunks[1].continuation_flag, "True")
    
    def test_post_process_all(self):
        """Test post-processing all chunks."""
        # Create test chunks
        chunks = [
            Chunk(
                id="chunk1",
                content="Content 1",
                heading_hierarchy=["Doc", "Section 1"],
                page_numbers=[1],
                continuation_flag="False",
                source_batch=0,
                metadata={"position_in_batch": 0}
            ),
            Chunk(
                id="chunk2",
                content="Content 2",
                heading_hierarchy=["Doc", "Section 1"],
                page_numbers=[1],
                continuation_flag="True",  # This should be merged with the previous chunk
                source_batch=0,
                metadata={"position_in_batch": 1}
            ),
            Chunk(
                id="chunk3",
                content="Content 3",
                heading_hierarchy=["Doc", "Section 2"],
                page_numbers=[2],
                continuation_flag="False",
                source_batch=0,
                metadata={"position_in_batch": 2}
            )
        ]
        
        processor = ChunkProcessor()
        merged_chunks = processor.post_process_all(chunks)
        
        # Check the merged chunks
        self.assertEqual(len(merged_chunks), 2)  # Two chunks after merging
        self.assertTrue("Content 1" in merged_chunks[0].content)
        self.assertTrue("Content 2" in merged_chunks[0].content)
        self.assertTrue("Content 3" in merged_chunks[1].content)


@pytest.mark.skip(reason="Requires API keys and real PDF")
class TestVisionChunker(unittest.TestCase):
    """Integration tests for the VisionChunker class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Set environment variables
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
        
        # Create a mock PDF extractor
        self.pdf_extractor_patcher = patch("src.utils.pdf.PDFExtractor")
        self.mock_pdf_extractor = self.pdf_extractor_patcher.start()
        
        # Create a mock LMM client
        self.lmm_client_patcher = patch("src.lmm_client.LMMClient")
        self.mock_lmm_client = self.lmm_client_patcher.start()
        
        # Configure the mock LMM client
        mock_instance = self.mock_lmm_client.return_value
        mock_instance.process_batch.return_value = """
[CHUNK]
[HEADING_HIERARCHY]Document > Chapter 1 > Section 1.1[/HEADING_HIERARCHY]
[CONTENT]
This is the content of the first chunk.
[/CONTENT]
[CONTINUES]False[/CONTINUES]
[/CHUNK]
"""
    
    def tearDown(self):
        """Clean up after the test."""
        self.pdf_extractor_patcher.stop()
        self.lmm_client_patcher.stop()
    
    def test_process_document(self):
        """Test processing a document."""
        # Configure the mock PDF extractor
        mock_extractor_instance = self.mock_pdf_extractor.return_value
        mock_extractor_instance.extract_page_images.return_value = [
            PageImage(image=None, page_number=1, width=100, height=100)
        ]
        
        # Create the chunker
        chunker = VisionChunker(batch_size=1)
        
        # Process a document
        chunks = chunker.process_document("test.pdf")
        
        # Check the result
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].heading_hierarchy, ["Document", "Chapter 1", "Section 1.1"])
        self.assertEqual(chunks[0].content, "This is the content of the first chunk.")


if __name__ == "__main__":
    unittest.main()
