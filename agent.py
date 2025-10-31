"""
LangChain agent for document analysis using Ollama.
"""
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import Dict, Any

from database_services import SQLiteService, MilvusService
from tools import TOOLS, init_tools
from config import OllamaConfig, DatabaseConfig, AGENT_MAX_ITERATIONS, AGENT_VERBOSE


class DocumentAnalysisAgent:
    """
    LangChain-based agent for document analysis.
    Uses Ollama LLM and ReAct framework.
    """
    
    def __init__(
        self,
        ollama_config: OllamaConfig,
        db_config: DatabaseConfig
    ):
        """
        Initialize the document analysis agent.
        
        Args:
            ollama_config: Ollama configuration
            db_config: Database configuration
        """
        self.ollama_config = ollama_config
        self.db_config = db_config
        
        # Initialize database services
        self.sqlite_service = SQLiteService(db_config.sqlite_path)
        self.milvus_service = MilvusService(
            db_config.milvus_host,
            db_config.milvus_port,
            db_config.collection_name
        )
        
        # Initialize tools with services
        init_tools(self.sqlite_service, self.milvus_service)
        
        # Initialize LLM
        self.llm = Ollama(
            model=ollama_config.model,
            temperature=ollama_config.temperature,
            base_url=ollama_config.base_url
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain ReAct agent."""
        
        # Define the prompt template for ReAct
        template = """You are a document analysis assistant with access to tools for analyzing documents stored in a database.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important guidelines:
- Always use tools to get factual information about documents
- Parse JSON responses from tools carefully
- Provide clear, concise answers based on the tool outputs
- If a tool returns an error, try a different approach or explain the limitation

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Create ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=TOOLS,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=AGENT_VERBOSE,
            max_iterations=AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a user query using the agent.
        
        Args:
            question: User's question about documents
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            result = self.agent_executor.invoke({"input": question})
            
            return {
                "success": True,
                "question": question,
                "answer": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "error": str(e)
            }
    
    def get_available_tools(self) -> list:
        """Get list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args": tool.args
            }
            for tool in TOOLS
        ]


class DocumentAnalysisAgentWithMemory(DocumentAnalysisAgent):
    """
    Agent with conversation memory for follow-up questions.
    """
    
    def __init__(
        self,
        ollama_config: OllamaConfig,
        db_config: DatabaseConfig
    ):
        """Initialize agent with memory."""
        super().__init__(ollama_config, db_config)
        
        # Add memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Create agent with memory support."""
        
        template = """You are a document analysis assistant with access to tools for analyzing documents.

Previous conversation:
{chat_history}

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
        
        prompt = PromptTemplate.from_template(template)
        
        agent = create_react_agent(
            llm=self.llm,
            tools=TOOLS,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            memory=self.memory,
            verbose=AGENT_VERBOSE,
            max_iterations=AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
