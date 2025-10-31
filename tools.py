"""
LangChain tools for document analysis.
Each tool is a function that the agent can use.
"""
from langchain.tools import tool
from typing import Optional, List
from datetime import datetime
import json

from database_services import SQLiteService, MilvusService



# Global services (will be initialized by factory)
_sqlite_service: Optional[SQLiteService] = None
_milvus_service: Optional[MilvusService] = None


def init_tools(sqlite_service: SQLiteService, milvus_service: MilvusService):
    """Initialize the tools with database services."""
    global _sqlite_service, _milvus_service
    _sqlite_service = sqlite_service
    _milvus_service = milvus_service


@tool
def get_documents_by_time_period(start_date: str, end_date: str) -> str:
    """
    Retrieve all documents from a specific time period.
    
    Args:
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
        
    Returns:
        JSON string with documents and their metadata
    """
    if not _sqlite_service:
        return json.dumps({"error": "Database service not initialized"})
    
    try:
        documents = _sqlite_service.get_documents_by_time_period(start_date, end_date)
        
        # Enrich with entities
        for doc in documents:
            doc['entities'] = _sqlite_service.get_entities_for_document(doc['id'])
        
        result = {
            "success": True,
            "count": len(documents),
            "time_period": f"{start_date} to {end_date}",
            "documents": documents[:10]  # Limit for context
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e)})




@tool
def analyze_document_patterns(classification: Optional[str] = None) -> str:
    """
    Analyze patterns across documents including temporal, entity, and classification patterns.
    
    Args:
        classification: Optional filter by document type (email, report, scientific)
        
    Returns:
        JSON string with pattern analysis results
    """
    if not _sqlite_service:
        return json.dumps({"error": "Database service not initialized"})
    
    try:
        # Get documents
        if classification:
            documents = _sqlite_service.get_documents_by_classification(classification)
        else:
            documents = _sqlite_service.get_all_documents()
        
        # Enrich with entities
        for doc in documents:
            doc['entities'] = _sqlite_service.get_entities_for_document(doc['id'])
        
        # Analyze patterns
        patterns = _analyze_patterns(documents)
        
        result = {
            "success": True,
            "document_count": len(documents),
            "patterns": patterns
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def find_duplicate_documents(
    classification: Optional[str] = None,
    similarity_threshold: float = 0.85
) -> str:
    """
    Find duplicate or near-duplicate documents using vector similarity.
    
    Args:
        classification: Optional filter by document type
        similarity_threshold: Similarity threshold (0.0 to 1.0, default 0.85)
        
    Returns:
        JSON string with duplicate groups and similarity scores
    """
    if not _sqlite_service or not _milvus_service:
        return json.dumps({"error": "Database services not initialized"})
    
    try:
        # Get documents
        if classification:
            documents = _sqlite_service.get_documents_by_classification(classification)
        else:
            documents = _sqlite_service.get_all_documents()
        
        if len(documents) < 2:
            return json.dumps({
                "success": True,
                "duplicate_groups": [],
                "message": "Need at least 2 documents to find duplicates"
            })
        
        # Get document IDs
        doc_ids = [doc['id'] for doc in documents]
        
        # Compute similarity matrix
        similarity_matrix = _milvus_service.compute_similarity_matrix(doc_ids)
        
        # Find duplicates
        duplicate_groups = []
        processed_ids = set()
        
        for doc in documents:
            doc_id = doc['id']
            
            if doc_id in processed_ids:
                continue
            
            similar_docs = similarity_matrix.get(doc_id, [])
            duplicates = []
            
            for similar in similar_docs:
                similarity_score = 1 - similar['distance']
                
                if similarity_score >= similarity_threshold and similar['id'] not in processed_ids:
                    duplicates.append({
                        "document_id": similar['id'],
                        "similarity_score": float(similarity_score)
                    })
            
            if duplicates:
                processed_ids.add(doc_id)
                for dup in duplicates:
                    processed_ids.add(dup['document_id'])
                
                duplicate_groups.append({
                    "primary_document_id": doc_id,
                    "duplicates": duplicates,
                    "group_size": len(duplicates) + 1
                })
        
        result = {
            "success": True,
            "total_documents": len(documents),
            "duplicate_groups": duplicate_groups[:10],  # Limit for context
            "total_groups": len(duplicate_groups)
        }
        
        return json.dumps(result, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


# Helper functions

def _analyze_patterns(documents: list) -> dict:
    """Analyze patterns across documents."""
    from collections import Counter
    
    # Classification distribution
    classifications = [doc.get('classification', 'unknown') for doc in documents]
    classification_dist = dict(Counter(classifications))
    
    # Entity patterns
    all_entities = []
    for doc in documents:
        all_entities.extend(doc.get('entities', []))
    
    entity_labels = [e.get('label') for e in all_entities]
    entity_dist = dict(Counter(entity_labels))
    
    # Temporal patterns
    dates = []
    for doc in documents:
        date_str = doc.get('date')
        if date_str:
            try:
                dates.append(datetime.fromisoformat(date_str))
            except:
                pass
    
    temporal_info = {}
    if dates:
        dates.sort()
        temporal_info = {
            "earliest_date": dates[0].isoformat(),
            "latest_date": dates[-1].isoformat(),
            "total_span_days": (dates[-1] - dates[0]).days
        }
    
    return {
        "classification_distribution": classification_dist,
        "entity_distribution": entity_dist,
        "temporal_patterns": temporal_info,
        "total_entities": len(all_entities)
    }


# Export tools list for LangChain agent
TOOLS = [
    get_documents_by_time_period,
    analyze_document_patterns,
    find_duplicate_documents
]
