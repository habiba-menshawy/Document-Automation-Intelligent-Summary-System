"""
Factory for creating document analysis agents.
Follows Dependency Inversion Principle.
"""
from agent import DocumentAnalysisAgent, DocumentAnalysisAgentWithMemory
from config import OllamaConfig, DatabaseConfig


class AgentFactory:
    """Factory for creating document analysis agents."""
    
    @staticmethod
    def create_agent(
        ollama_config: OllamaConfig = None,
        db_config: DatabaseConfig = None,
        with_memory: bool = False
    ):
        """
        Create a document analysis agent.
        
        Args:
            ollama_config: Optional Ollama configuration (uses defaults if None)
            db_config: Optional database configuration (uses defaults if None)
            with_memory: Whether to create agent with conversation memory
            
        Returns:
            DocumentAnalysisAgent or DocumentAnalysisAgentWithMemory instance
        """
        # Use defaults if not provided
        if ollama_config is None:
            ollama_config = OllamaConfig()
        
        if db_config is None:
            db_config = DatabaseConfig()
        
        # Create appropriate agent
        if with_memory:
            return DocumentAnalysisAgentWithMemory(ollama_config, db_config)
        else:
            return DocumentAnalysisAgent(ollama_config, db_config)
    
    @staticmethod
    def create_custom_agent(
        model: str,
        db_path: str,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        with_memory: bool = False
    ):
        """
        Create an agent with custom configuration.
        
        Args:
            model: Ollama model name
            db_path: Path to SQLite database
            milvus_host: Milvus host
            milvus_port: Milvus port
            with_memory: Whether to use conversation memory
            
        Returns:
            Configured agent instance
        """
        ollama_config = OllamaConfig(model=model)
        db_config = DatabaseConfig(
            sqlite_path=db_path,
            milvus_host=milvus_host,
            milvus_port=milvus_port
        )
        
        return AgentFactory.create_agent(
            ollama_config,
            db_config,
            with_memory=with_memory
        )
