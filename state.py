from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    """The state of the agent graph."""
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    user_id: str
    agent_id: str
    
    # Analysis & Context
    intent: Dict[str, Any] 
    rag_data: Dict[str, Any] 
    skill_results: Dict[str, Any] 
    
    # Metadata
    user_info: Dict[str, Any] 
    agent_config: Dict[str, Any] 
    # skills_map removed
