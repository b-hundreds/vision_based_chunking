"""
Context management module for preserving context between document batches.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Context:
    """Context information for batch processing."""
    
    summary: str = ""
    last_chunk: str = ""
    heading_hierarchy: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextManager:
    """
    Manages context information between batches of document pages.
    
    This component is responsible for:
    1. Creating the initial context for the first batch
    2. Updating the context after each batch is processed
    3. Extracting context information from chunks
    """
    
    def create_initial_context(self) -> Context:
        """
        Create initial context for the first batch.
        
        Returns:
            Empty context object
        """
        return Context(
            summary="",
            last_chunk="",
            heading_hierarchy=[],
            metadata={},
        )
    
    def update_context(
        self,
        current_chunks: List[Any],  # List[Chunk]
        previous_context: Context,
    ) -> Context:
        """
        Update context based on current batch processing results.
        
        Args:
            current_chunks: Chunks from the current batch
            previous_context: Context from the previous batch
            
        Returns:
            Updated context for the next batch
        """
        if not current_chunks:
            return previous_context
            
        # Get the last chunk from the current batch
        last_chunk = current_chunks[-1]
        
        # Create a summary of the current batch
        # This is a simplified implementation; in practice, you might use an LLM to create a summary
        summary = self._create_batch_summary(current_chunks)
        
        # Extract the latest heading hierarchy
        heading_hierarchy = last_chunk.heading_hierarchy
        
        # Create updated context
        return Context(
            summary=summary,
            last_chunk=last_chunk.content,
            heading_hierarchy=heading_hierarchy,
            metadata={
                "last_page_processed": max(chunk.page_numbers for chunk in current_chunks),
                "document_structure": self._extract_document_structure(current_chunks),
            },
        )
    
    def _create_batch_summary(self, chunks: List[Any]) -> str:
        """
        Create a summary of the current batch.
        
        Args:
            chunks: List of chunks from the current batch
            
        Returns:
            Summary text
        """
        # In a real implementation, you might use an LLM to generate a summary
        # For simplicity, we'll just concatenate headings
        headings = set()
        for chunk in chunks:
            if chunk.heading_hierarchy:
                headings.add(" > ".join(chunk.heading_hierarchy))
        
        if not headings:
            return "No headings found in this batch."
            
        return f"This batch contains content from the following sections: {'; '.join(headings)}"
    
    def _extract_document_structure(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Extract document structure information from the current batch.
        
        Args:
            chunks: List of chunks from the current batch
            
        Returns:
            Document structure metadata
        """
        # This is a simplified implementation
        # In practice, you would build a more sophisticated document structure model
        
        # Extract heading levels
        heading_levels = {}
        for chunk in chunks:
            if not chunk.heading_hierarchy:
                continue
                
            for i, heading in enumerate(chunk.heading_hierarchy):
                level = i + 1
                if level not in heading_levels:
                    heading_levels[level] = set()
                heading_levels[level].add(heading)
        
        # Convert sets to lists for JSON serialization
        structure = {
            f"level_{level}": list(headings)
            for level, headings in heading_levels.items()
        }
        
        return structure
