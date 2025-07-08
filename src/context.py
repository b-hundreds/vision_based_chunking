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
        
        # Extract the latest heading hierarchy, but with some intelligence
        heading_hierarchy = self._determine_heading_hierarchy(current_chunks, previous_context)
        
        # Get the highest page number from this batch
        max_page = max(max(chunk.page_numbers) for chunk in current_chunks)
        
        # Create updated context
        return Context(
            summary=summary,
            last_chunk=last_chunk.content,
            heading_hierarchy=heading_hierarchy,
            metadata={
                "last_page_processed": max_page,
                "document_structure": self._extract_document_structure(current_chunks),
                "current_page_range": [min(min(chunk.page_numbers) for chunk in current_chunks), max_page],
            },
        )
        
    def _determine_heading_hierarchy(self, current_chunks: List[Any], previous_context: Context) -> List[str]:
        """
        Intelligently determine the appropriate heading hierarchy for the next batch.
        
        This method analyzes the current batch to determine if we should:
        1. Use a heading hierarchy from the current batch
        2. Carry over the hierarchy from the previous batch
        3. Reset the hierarchy because we've moved to a new section
        
        Args:
            current_chunks: Chunks from the current batch
            previous_context: Context from the previous batch
            
        Returns:
            Appropriate heading hierarchy for the next batch
        """
        # If the current batch has no chunks, keep the previous hierarchy
        if not current_chunks:
            return previous_context.heading_hierarchy
            
        # Check if any chunk in this batch has a complete (non-continuation) heading hierarchy
        complete_hierarchies = [
            chunk.heading_hierarchy
            for chunk in current_chunks 
            if chunk.heading_hierarchy and chunk.continuation_flag != "True"
        ]
        
        # If we have complete hierarchies in this batch, use the last one
        if complete_hierarchies:
            return complete_hierarchies[-1]
        
        # If we have any hierarchies in this batch, prefer the most specific one
        all_hierarchies = [chunk.heading_hierarchy for chunk in current_chunks if chunk.heading_hierarchy]
        if all_hierarchies:
            # Choose the most specific (longest) hierarchy
            return max(all_hierarchies, key=len)
            
        # If we're still here, we need to decide if we should keep the previous hierarchy
        # Check if the pages in this batch are continuous with the previous batch
        if previous_context.metadata and "last_page_processed" in previous_context.metadata:
            last_page = previous_context.metadata["last_page_processed"]
            current_first_page = min(min(chunk.page_numbers) for chunk in current_chunks)
            
            # If there's a gap of more than 1 page, reset the hierarchy
            if current_first_page > last_page + 1:
                return []
                
        # If we're on consecutive pages, keep the previous hierarchy
        return previous_context.heading_hierarchy
    
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
