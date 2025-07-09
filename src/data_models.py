from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union


@dataclass
class RawChunk:
    """Raw chunk structure as parsed from the LLM response."""
    continues: Union[bool, Literal["partial"]]
    heading: str
    content: str


@dataclass
class FinalChunk:
    """Final processed chunk ready for output."""
    id: str
    heading: str
    content: str
    main_heading: str
    section_heading: str
    chunk_title: str
    page_numbers: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "heading": self.heading,
            "content": self.content,
            "main_heading": self.main_heading,
            "section_heading": self.section_heading,
            "chunk_title": self.chunk_title,
            "page_numbers": self.page_numbers
        }
