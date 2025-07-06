"""
Vector database storage module for chunk storage and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class VectorDB(ABC):
    """
    Abstract base class for vector database integration.
    
    This defines the interface for storing and retrieving chunks in a vector database.
    Concrete implementations should be provided for specific vector databases.
    """
    
    @classmethod
    def create(cls, db_type: str, config: Dict[str, Any]) -> 'VectorDB':
        """
        Factory method to create a VectorDB instance.
        
        Args:
            db_type: Type of vector database (e.g., 'elasticsearch')
            config: Configuration parameters for the database
            
        Returns:
            VectorDB instance
        """
        if db_type.lower() == 'elasticsearch':
            return ElasticsearchDB(config)
        else:
            raise ValueError(f"Unsupported vector database type: {db_type}")
    
    @abstractmethod
    def ingest_chunks(
        self,
        chunks: List[Any],  # List[Chunk]
        embedding_model: str,
    ) -> bool:
        """
        Embed and ingest chunks into the vector database.
        
        Args:
            chunks: List of chunks to ingest
            embedding_model: Name of the embedding model to use
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in the vector database.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_criteria: Additional filter criteria
            
        Returns:
            List of relevant chunks with metadata
        """
        pass


class ElasticsearchDB(VectorDB):
    """
    Elasticsearch implementation of the VectorDB interface.
    
    This class handles:
    1. Connection to Elasticsearch
    2. Creating and managing indices
    3. Embedding and storing chunks
    4. Searching for chunks using vector similarity
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Elasticsearch client.
        
        Args:
            config: Configuration parameters for Elasticsearch
                host: Elasticsearch host
                port: Elasticsearch port
                username: Elasticsearch username (optional)
                password: Elasticsearch password (optional)
                index: Elasticsearch index name
        """
        self.config = config
        self.index_name = config.get('index', 'vision_chunks')
        self._init_client()
        self._init_index()
    
    def _init_client(self):
        """Initialize the Elasticsearch client."""
        try:
            from elasticsearch import Elasticsearch
            
            # Create client with authentication if provided
            if self.config.get('username') and self.config.get('password'):
                self.client = Elasticsearch(
                    [f"{self.config['host']}:{self.config['port']}"],
                    basic_auth=(self.config['username'], self.config['password'])
                )
            else:
                self.client = Elasticsearch(
                    [f"{self.config['host']}:{self.config['port']}"]
                )
                
            logger.info(f"Connected to Elasticsearch at {self.config['host']}:{self.config['port']}")
            
        except ImportError:
            raise ImportError(
                "elasticsearch package not installed. "
                "Install it with: pip install elasticsearch"
            )
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {str(e)}")
            raise
    
    def _init_index(self):
        """Initialize the Elasticsearch index if it doesn't exist."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                # Create index with appropriate mappings for vector search
                mappings = {
                    "properties": {
                        "id": {"type": "keyword"},
                        "content": {"type": "text"},
                        "heading_hierarchy": {"type": "keyword"},
                        "page_numbers": {"type": "integer"},
                        "continuation_flag": {"type": "keyword"},
                        "source_batch": {"type": "integer"},
                        "metadata": {
                            "type": "object",
                            "enabled": True
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,  # Default for text-embedding-3-small
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
                
                self.client.indices.create(
                    index=self.index_name,
                    mappings=mappings
                )
                
                logger.info(f"Created Elasticsearch index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error creating Elasticsearch index: {str(e)}")
            raise
    
    def ingest_chunks(
        self,
        chunks: List[Any],  # List[Chunk]
        embedding_model: str,
    ) -> bool:
        """
        Embed and ingest chunks into Elasticsearch.
        
        Args:
            chunks: List of chunks to ingest
            embedding_model: Name of the embedding model to use
            
        Returns:
            Success status
        """
        try:
            # Get embeddings for chunks
            embeddings = self._get_embeddings(
                texts=[chunk.content for chunk in chunks],
                model=embedding_model
            )
            
            # Prepare documents for bulk indexing
            bulk_data = []
            for i, chunk in enumerate(chunks):
                # Convert dataclass to dict
                chunk_dict = {
                    "id": chunk.id,
                    "content": chunk.content,
                    "heading_hierarchy": chunk.heading_hierarchy,
                    "page_numbers": chunk.page_numbers,
                    "continuation_flag": chunk.continuation_flag,
                    "source_batch": chunk.source_batch,
                    "metadata": chunk.metadata,
                    "embedding": embeddings[i]
                }
                
                # Add index operation
                bulk_data.append({"index": {"_index": self.index_name, "_id": chunk.id}})
                bulk_data.append(chunk_dict)
            
            # Execute bulk operation
            if bulk_data:
                self.client.bulk(operations=bulk_data, refresh=True)
                logger.info(f"Ingested {len(chunks)} chunks into Elasticsearch")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error ingesting chunks into Elasticsearch: {str(e)}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks in Elasticsearch using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_criteria: Additional filter criteria
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Get embedding for query
            query_embedding = self._get_embeddings([query], "text-embedding-3-small")[0]
            
            # Build search query
            search_query = {
                "size": top_k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                }
            }
            
            # Add filter if provided
            if filter_criteria:
                search_query["query"] = {
                    "bool": {
                        "must": search_query["query"],
                        "filter": self._build_filter(filter_criteria)
                    }
                }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=search_query
            )
            
            # Extract results
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                # Remove embedding from result
                if "embedding" in source:
                    del source["embedding"]
                source["score"] = hit["_score"]
                results.append(source)
                
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {str(e)}")
            return []
    
    def _get_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            model: Name of the embedding model
            
        Returns:
            List of embeddings
        """
        # Use OpenAI for text-embedding models
        if model.startswith("text-embedding-"):
            return self._get_openai_embeddings(texts, model)
        else:
            raise ValueError(f"Unsupported embedding model: {model}")
    
    def _get_openai_embeddings(self, texts: List[str], model: str) -> List[List[float]]:
        """
        Get embeddings using OpenAI's API.
        
        Args:
            texts: List of texts to embed
            model: Name of the OpenAI embedding model
            
        Returns:
            List of embeddings
        """
        try:
            import os
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            client = OpenAI(api_key=api_key)
            
            embeddings = []
            
            # Process in batches to avoid token limits
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            return embeddings
            
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Install it with: pip install openai"
            )
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {str(e)}")
            raise
    
    def _build_filter(self, filter_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Elasticsearch filter from filter criteria.
        
        Args:
            filter_criteria: Filter criteria
            
        Returns:
            Elasticsearch filter
        """
        filters = []
        
        # Handle common filter types
        for field, value in filter_criteria.items():
            if field.startswith("metadata."):
                # Handle metadata fields
                filters.append({"term": {field: value}})
            elif field == "page_numbers":
                # Handle page number range
                if isinstance(value, list) and len(value) == 2:
                    filters.append({
                        "range": {
                            field: {
                                "gte": value[0],
                                "lte": value[1]
                            }
                        }
                    })
                else:
                    filters.append({"term": {field: value}})
            else:
                # Handle regular fields
                filters.append({"term": {field: value}})
                
        return {"bool": {"must": filters}}
