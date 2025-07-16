import logging
import uuid
import re
from typing import List, Dict, Tuple

from src.data_models import RawChunk, FinalChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Handles post-processing of chunks, including merging continuing chunks and merging table chunks.
    """
    
    def __init__(self):
        """Initialize the post processor."""
        pass
    
    def _is_table_chunk(self, content: str) -> bool:
        """
        Check if a chunk contains table content based on table syntax (pipes | and hyphens -).
        
        Args:
            content: The content to check
            
        Returns:
            True if the content appears to be a table, False otherwise
        """
        lines = content.strip().split('\n')
        table_indicators = 0
        
        for line in lines:
            # Check for table syntax: lines with pipes | and header separators with hyphens -
            if '|' in line:
                table_indicators += 1
            # Check for header separator lines (containing hyphens and pipes)
            if re.search(r'^\s*\|[\s\-\|]+\|\s*$', line):
                table_indicators += 2  # Weight header separators more heavily
                
        # Consider it a table if we have enough table indicators
        return table_indicators >= 2
    
    def _extract_table_header(self, content: str) -> Tuple[str, str]:
        """
        Extract the table header from table content.
        
        Args:
            content: Table content
            
        Returns:
            Tuple of (header_lines, remaining_content)
        """
        lines = content.strip().split('\n')
        header_lines = []
        remaining_lines = []
        header_found = False
        
        for i, line in enumerate(lines):
            # Look for header separator line (contains hyphens and pipes)
            if re.search(r'^\s*\|[\s\-\|]+\|\s*$', line):
                # Include the header row and separator
                if i > 0:
                    header_lines.append(lines[i-1])  # Previous line is the header
                header_lines.append(line)  # Current line is the separator
                header_found = True
                # Everything after this is data rows
                remaining_lines = lines[i+1:]
                break
        
        # If no separator found but we have pipe-separated lines, 
        # assume first line is header and create a separator
        if not header_found and lines and '|' in lines[0]:
            header_line = lines[0]
            header_lines = [header_line]
            
            # Create a separator line based on the header
            # Count the number of columns by counting pipes
            pipe_count = header_line.count('|')
            if header_line.startswith('|') and header_line.endswith('|'):
                # Format: |col1|col2|col3| - columns = pipes - 1
                columns = pipe_count - 1
                separator = '|' + '|'.join(['-------' for _ in range(columns)]) + '|'
            else:
                # Format: col1|col2|col3 - columns = pipes + 1
                columns = pipe_count + 1
                separator = '|'.join(['-------' for _ in range(columns)])
            
            header_lines.append(separator)
            remaining_lines = lines[1:]
        
        return '\n'.join(header_lines), '\n'.join(remaining_lines)
    
    def _merge_table_chunks(self, chunks: List[FinalChunk]) -> List[FinalChunk]:
        """
        Merge table chunks that have the same second-level heading.
        
        Args:
            chunks: List of FinalChunk objects
            
        Returns:
            List of chunks with table chunks merged
        """
        merged_chunks = []
        table_groups = {}  # Key: section_heading, Value: list of table chunks
        
        for chunk in chunks:
            if self._is_table_chunk(chunk.content):
                # Group table chunks by section heading
                section_key = chunk.section_heading
                if section_key not in table_groups:
                    table_groups[section_key] = []
                table_groups[section_key].append(chunk)
            else:
                merged_chunks.append(chunk)
        
        # Process each table group
        for section_heading, table_chunks in table_groups.items():
            if len(table_chunks) == 1:
                # Only one table chunk, no merging needed
                merged_chunks.append(table_chunks[0])
            else:
                # Merge multiple table chunks
                logger.info(f"Merging {len(table_chunks)} table chunks for section: {section_heading}")
                
                # Use the first chunk as base
                base_chunk = table_chunks[0]
                merged_content_parts = []
                merged_page_numbers = list(base_chunk.page_numbers)
                
                # Extract header from first chunk
                header, first_data = self._extract_table_header(base_chunk.content)
                all_data_rows = []
                
                # Add data from first chunk
                if first_data.strip():
                    all_data_rows.extend([line for line in first_data.split('\n') if line.strip()])
                
                # Add data from remaining chunks (without their headers)
                for chunk in table_chunks[1:]:
                    _, data_only = self._extract_table_header(chunk.content)
                    if data_only.strip():
                        all_data_rows.extend([line for line in data_only.split('\n') if line.strip()])
                    
                    # Merge page numbers
                    for page_num in chunk.page_numbers:
                        if page_num not in merged_page_numbers:
                            merged_page_numbers.append(page_num)
                
                # Combine header and all data rows
                merged_content_parts = []
                if header:
                    merged_content_parts.append(header)
                if all_data_rows:
                    merged_content_parts.extend(all_data_rows)
                
                # Create merged chunk
                merged_chunk = FinalChunk(
                    id=str(uuid.uuid4()),
                    heading=base_chunk.heading,
                    content='\n'.join(merged_content_parts),
                    main_heading=base_chunk.main_heading,
                    section_heading=base_chunk.section_heading,
                    chunk_title=f"{base_chunk.chunk_title} (Merged Table)",
                    continues=base_chunk.continues,
                    page_numbers=sorted(merged_page_numbers)
                )
                
                merged_chunks.append(merged_chunk)
                logger.debug(f"Created merged table chunk: {merged_chunk.id}")
        
        return merged_chunks
    
    def process_chunks(self, raw_chunks: List[RawChunk], page_numbers: List[int]) -> List[FinalChunk]:
        """
        Process a list of raw chunks into final chunks, merging continuing chunks and table chunks.
        
        Args:
            raw_chunks: List of RawChunk objects to process
            page_numbers: List of page numbers corresponding to the batch
            
        Returns:
            List of processed FinalChunk objects
        """
        logger.info(f"Post-processing {len(raw_chunks)} chunks")
        
        final_chunks = []
        last_chunk = None
        
        for i, chunk in enumerate(raw_chunks):
            # Extract heading components
            heading_parts = chunk.heading.split(" > ")
            
            if len(heading_parts) != 3:
                logger.warning(f"Chunk heading '{chunk.heading}' does not have 3 parts. Using defaults.")
                # Use defaults if the heading doesn't have 3 parts
                main_heading = heading_parts[0] if heading_parts else "Unknown Main Heading"
                section_heading = heading_parts[1] if len(heading_parts) > 1 else "Unknown Section"
                chunk_title = heading_parts[2] if len(heading_parts) > 2 else "Unknown Chunk"
            else:
                main_heading = heading_parts[0]
                section_heading = heading_parts[1]
                chunk_title = heading_parts[2]
            
            # Handle continuation chunks
            if chunk.continues and last_chunk is not None:
                # Merge this chunk with the last one
                last_chunk.content += f"\n{chunk.content}"
                
                # Add page numbers
                # Estimate which page this chunk is from based on its position in the batch
                chunk_page = page_numbers[min(i, len(page_numbers) - 1)]
                if chunk_page not in last_chunk.page_numbers:
                    last_chunk.page_numbers.append(chunk_page)
                
                logger.debug(f"Merged continuing chunk into previous chunk: {last_chunk.id}")
            else:
                # Create a new final chunk
                new_chunk = FinalChunk(
                    id=str(uuid.uuid4()),
                    heading=chunk.heading,
                    content=chunk.content,
                    main_heading=main_heading,
                    section_heading=section_heading,
                    chunk_title=chunk_title,
                    continues=chunk.continues,
                    page_numbers=[page_numbers[min(i, len(page_numbers) - 1)]]
                )
                final_chunks.append(new_chunk)
                last_chunk = new_chunk
                logger.debug(f"Created new chunk: {new_chunk.id}")
        
        # Merge table chunks with the same section heading
        final_chunks = self._merge_table_chunks(final_chunks)
        
        logger.info(f"Post-processing complete. Produced {len(final_chunks)} final chunks.")
        return final_chunks
    
    def extract_last_chunk_context(self, final_chunks: List[FinalChunk]) -> str:
        """
        Extract context information from the last chunk to provide to the next batch.
        
        Args:
            final_chunks: List of FinalChunk objects
            
        Returns:
            Context string for the next batch in LAST CHUNKS format
        """
        if not final_chunks:
            return ""
        
        last_chunk = final_chunks[-1]
        
        # Create LAST CHUNKS context with the new format requirements
        # Include the last chunk's full information for continuity
        context = (
            f"LAST CHUNKS:\n"
            f"[CONTINUES]{str(last_chunk.continues).title()}[/CONTINUES]"
            f"[HEAD]{last_chunk.heading}[/HEAD]\n"
            f"{last_chunk.content}\n\n"
            f"Note: Use this information only for heading inference and content continuity. "
            f"Do not include this content in new chunks unless it directly continues from an incomplete sentence or table row."
        )
        
        return context
