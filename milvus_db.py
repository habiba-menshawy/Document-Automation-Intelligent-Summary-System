
"""
Milvus vector database implementation.
Follows Single Responsibility Principle - handles only Milvus operations.
"""

from config import OllamaConfig
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, MilvusException
import os
from typing import List, Dict, Any, Optional
import numpy as np

from logger.logger_config import Logger
import time
import ollama

log = Logger.get_logger(__name__)


class MilvusDB():
    """
    Milvus database implementation with error handling and logging.
    Implements vector database, insert, and search interfaces.
    """

    # def __init__(self, milvus_url: Optional[str] = "http://milvus-standalone:19530"):
    def __init__(self, milvus_url: Optional[str] = "http://localhost:19530"):
        """
        Initialize Milvus client.
        
        Args:
            milvus_url: Milvus server URL. If None, reads from MILVUS_URL env variable.
            
        Raises:
            ValueError: If MILVUS_URL is not set or invalid.
            Exception: If connection fails.
        """
        # Get URL from parameter or environment
        url =  os.getenv("MILVUS_URL") or milvus_url 
        
        if not url:
            log.error("Environment variable MILVUS_URL is not set.")
            raise ValueError("Environment variable MILVUS_URL is not set.")
        
        try:
            self.client = MilvusClient(url)
            log.info(f"Milvus client initialized successfully with URL: {url}")
        except ValueError as e:
            log.error(f"Initialization error: {e}")
            self.client = None
            raise ValueError(f"Failed to initialize Milvus client: {e}")
        except Exception as e:
            log.error(f"Failed to initialize Milvus client: {e}")
            self.client = None
            raise Exception(f"Failed to initialize Milvus client: {e}")
    
    def _validate_client(self) -> None:
        """Validate that client is initialized."""
        if self.client is None:
            raise AttributeError("Milvus client is not initialized.")
    
    def create_indexing(self, field_name: str = "embedding", 
                       index_type: str = "FLAT", 
                       metric_type: str = "COSINE",
                       params: Optional[Dict[str, Any]] = None):
        """
        Create index parameters for a collection.
        
        Args:
            field_name: Name of the field to index.
            index_type: Type of index (FLAT, IVF_FLAT, HNSW, etc.).
            metric_type: Metric type (COSINE, IP, L2).
            params: Additional index parameters.
            
        Returns:
            Index parameters object.
            
        Raises:
            AttributeError: If client is not initialized.
            Exception: If index creation fails.
        """
        try:
            self._validate_client()
            
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params or {"nlist": 512}
            )
            
            log.info(f"Index parameters created: {index_type} with {metric_type}")
            return index_params
            
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error creating indexing parameters: {e}")
            raise Exception(f"Error creating indexing parameters: {e}")
    
    def create_fields(self, embedding_dim: int = 768, include_text: bool = True) -> List[FieldSchema]:
        """
        Create field schema for collection.
        
        Args:
            embedding_dim: Dimension of embedding vectors.
            include_text: Whether to include text field.
            
        Returns:
            List of FieldSchema objects.
            
        Raises:
            Exception: If field creation fails.
        """
        try:
            fields = [
                FieldSchema(
                    name="id", 
                    dtype=DataType.INT64, 
                    is_primary=True, 
                    auto_id=True,
                    description="Primary ID"
                ),
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=embedding_dim,
                    description="Embedding vector"
                ),
                FieldSchema(
                    name="created_at", 
                    dtype=DataType.INT64,
                    description="UNIX timestamp"
                )
            ]
            
            if include_text:
                fields.insert(1, FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                    description="Document text"
                ))
            
            log.info(f"Field schema created with {len(fields)} fields")
            return fields
            
        except Exception as e:
            log.error(f"Error creating fields schema: {e}")
            raise Exception(f"Error creating fields schema: {e}")
    
    def create_schema(self, fields: List[FieldSchema], 
                     enable_dynamic: bool = True,
                     description: str = "") -> CollectionSchema:
        """
        Create collection schema.
        
        Args:
            fields: List of field schemas.
            enable_dynamic: Enable dynamic fields.
            description: Schema description.
            
        Returns:
            CollectionSchema object.
            
        Raises:
            Exception: If schema creation fails.
        """
        try:
            schema = CollectionSchema(
                fields=fields,
                enable_dynamic_field=enable_dynamic,
                description=description
            )
            log.info(f"Collection schema created: {description}")
            return schema
            
        except Exception as e:
            log.error(f"Error creating collection schema: {e}")
            raise Exception(f"Error creating collection schema: {e}")
        

    def create_db(self, collection_name: str) -> dict:
        """Create a new database collection in Milvus."""
        # Setup milvus DB
        try:
            fields = self.create_fields()  # Create fields for the collection
            schema = self.create_schema(fields)  # Create schema for the collection
            index_params = self.create_indexing()  # Create index parameters
            self.create_collection(schema=schema, collection_name=collection_name, index_params=index_params)  # Create collection in Milvus
            log.info(f"Database '{collection_name}' created successfully.")
            return {'collection_name': collection_name}
        except Exception as e:
            # Handle exceptions and return an error response
            log.error(f"Failed to create database: {e}")
            raise Exception(f"Error creating database: {e}")
        
    def get_row_count(self, collection_name: str) -> int:
        """
        Get the number of rows in a collection.
        
        Args:
            collection_name: Name of the collection.
            
        Returns:
            Number of rows.
            
        Raises:
            AttributeError: If client is not initialized.
            Exception: If retrieval fails.
        """
        try:
            self._validate_client()
            
            result = self.client.get_collection_stats(collection_name)
            row_count = result.get('row_count', 0)
            
            log.info(f"Collection '{collection_name}' has {row_count} rows")
            return row_count
            
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error retrieving row count for '{collection_name}': {e}")
            raise Exception(f"Error retrieving row count for '{collection_name}': {e}")
    
    # IVectorDatabase implementation
    
    def create_collection(self, collection_name: str, schema: CollectionSchema, 
                         index_params: Any) -> None:
        """
        Create a collection with schema and index.
        
        Args:
            collection_name: Name of the collection.
            schema: Collection schema.
            index_params: Index parameters.
            
        Raises:
            AttributeError: If client is not initialized.
            Exception: If creation fails.
        """
        try:
            self._validate_client()
            
            # Check if collection exists
            if self.has_collection(collection_name):
                row_count = self.get_row_count(collection_name)
                
                if row_count > 0:
                    log.info(f"Collection '{collection_name}' exists with {row_count} rows. Skipping.")
                    return
                else:
                    log.info(f"Collection '{collection_name}' is empty. Dropping and recreating.")
                    self.drop_collection(collection_name)
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            
            collections = self.list_collections()
            log.info(f"Collection '{collection_name}' created. Total: {len(collections)}")
            
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error creating collection '{collection_name}': {e}")
            raise Exception(f"Error creating collection '{collection_name}': {e}")
    
    def has_collection(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            self._validate_client()
            return self.client.has_collection(collection_name=collection_name)
        except Exception as e:
            log.error(f"Error checking collection '{collection_name}': {e}")
            raise
    
    def drop_collection(self, collection_name: str) -> None:
        """Drop a collection."""
        try:
            self._validate_client()
            self.client.drop_collection(collection_name=collection_name)
            log.info(f"Collection '{collection_name}' dropped")
        except Exception as e:
            log.error(f"Error dropping collection '{collection_name}': {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        try:
            self._validate_client()
            return self.client.list_collections()
        except Exception as e:
            log.error(f"Error listing collections: {e}")
            raise
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            self._validate_client()
            return self.client.get_collection_stats(collection_name)
        except Exception as e:
            log.error(f"Error getting stats for '{collection_name}': {e}")
            raise
    
    # IVectorInsert implementation

    def insert(self, collection_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert data into collection.
        
        Args:
            collection_name: Name of the collection.
            data: List of data to insert [embeddings, texts, timestamps].
            
        Returns:
            Insert result with primary keys.
            
        Raises:
            TypeError: If data format is invalid.
            AttributeError: If client is not initialized.
            Exception: If insertion fails.
        """
        try:
            self._validate_client()
            
            result = self.client.insert(collection_name=collection_name, data=data)
            print(result)
            log.info(f"Inserted data into '{collection_name}' with {result['insert_count']} records")
            return result
            
        except TypeError as e:
            log.error("Invalid data format for insertion.")
            raise TypeError(f"Invalid data format: {e}")
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error inserting into '{collection_name}': {e}")
            raise Exception(f"Error inserting into '{collection_name}': {e}")
    
    def flush(self, collection_name: str) -> None:
        """Flush data to disk."""
        try:
            self._validate_client()
            self.client.flush(collection_name)
            log.info(f"Flushed collection '{collection_name}'")
        except Exception as e:
            log.error(f"Error flushing '{collection_name}': {e}")
            raise
    
    # IVectorSearch implementation
    
    def search(self, collection_name: str, query_vectors: List[List[float]], 
               top_k: int, search_params: Dict[str, Any],
               output_fields: Optional[List[str]] = None) -> List[Any]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Name of the collection.
            query_vectors: List of query vectors.
            top_k: Number of results to return.
            search_params: Search parameters.
            output_fields: Fields to return.
            
        Returns:
            Search results.
            
        Raises:
            ValueError: If parameters are invalid.
            AttributeError: If client is not initialized.
            Exception: If search fails.
        """
        try:
            self._validate_client()
            
            results = self.client.search(
                collection_name=collection_name,
                data=query_vectors,
                anns_field="embedding",
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields or ["text", "created_at"]
            )
            
            log.info(f"Search completed in '{collection_name}' with {len(results)} results")
            return results
            
        except ValueError as e:
            log.error("Invalid search parameters.")
            raise ValueError(f"Invalid search parameters: {e}")
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error searching in '{collection_name}': {e}")
            raise Exception(f"Error searching in '{collection_name}': {e}")
    
    def query(self, collection_name: str, filter_expr: str,
              output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Query collection with filter.
        
        Args:
            collection_name: Name of the collection.
            filter_expr: Filter expression.
            output_fields: Fields to return.
            
        Returns:
            Query results.
            
        Raises:
            ValueError: If query expression is invalid.
            AttributeError: If client is not initialized.
            Exception: If query fails.
        """
        try:
            self._validate_client()
            
            results = self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["embedding", "text", "created_at"]
            )
            
            log.info(f"Query completed in '{collection_name}' with {len(results)} results")
            return results
            
        except ValueError as e:
            log.error("Invalid query expression.")
            raise ValueError(f"Invalid query expression: {e}")
        except AttributeError as e:
            log.error("Milvus client is not initialized.")
            raise AttributeError(f"Milvus client is not initialized: {e}")
        except Exception as e:
            log.error(f"Error querying '{collection_name}': {e}")
            raise Exception(f"Error querying '{collection_name}': {e}")
    
    def get_database_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive summary of all collections.
        
        Returns:
            Dictionary with collection statistics.
            
        Raises:
            MilvusException: If retrieval fails.
        """
        database_summary = {}
        
        try:
            self._validate_client()
            
            collection_names = self.list_collections()
            log.info(f"Retrieved {len(collection_names)} collections")
            
            for collection_name in collection_names:
                try:
                    # Get row count
                    stats = self.get_collection_stats(collection_name)
                    num_entities = stats.get('row_count', 0)
                    
                    # Get schema
                    schema = self.client.describe_collection(collection_name)
                    fields = schema.get('fields', [])
                    
                    # Get index info
                    index_info = []
                    try:
                        index_names = self.client.list_indexes(collection_name)
                        for index_name in index_names:
                            try:
                                index_details = self.client.describe_index(
                                    collection_name, 
                                    index_name=index_name
                                )
                                index_info.append(index_details)
                            except MilvusException as e:
                                log.error(f"Error retrieving index '{index_name}': {e}")
                    except Exception as e:
                        log.warning(f"Could not retrieve indexes for '{collection_name}': {e}")
                    
                    database_summary[collection_name] = {
                        'num_rows': num_entities,
                        'fields': fields,
                        'index_info': index_info
                    }
                    
                    log.info(f"Collection '{collection_name}': {num_entities} rows")
                    
                except MilvusException as e:
                    log.error(f"Error retrieving info for '{collection_name}': {e}")
                    raise
            
            return database_summary
            
        except MilvusException as e:
            log.error(f"Failed to retrieve database summary: {e}")
            raise
        except Exception as e:
            log.error(f"Unexpected error in get_database_summary: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    milvus_db = MilvusDB()
    summary = milvus_db.create_db("example_collection")
    # 2. Generate embedding
    try:
        log.info("Generating embedding...")
        # embedding_vector = np.array(ollama.embed(model=OllamaConfig.embedding_model, input="any thing litteraly")['embeddings'][0])
    except Exception as e:
        log.error(f"Failed to generate embedding: {e}")
        embedding_vector = np.random.rand(768).tolist()

    # 3. Insert into Milvus first
    doc_id = None
    
    log.info("Inserting into Milvus...")
    milvus_data = {
        "embedding":  np.random.rand(768).tolist(),
        "created_at": int(time.time()),
        "text": "any thing litteraly"
    }
    insert_result = milvus_db.insert("example_collection", milvus_data)
    milvus_db.flush("example_collection")
    doc_id = insert_result.get('ids')[0]  # Get the Milvus PK
    log.info(f"Inserted into Milvus with PK={doc_id}")
