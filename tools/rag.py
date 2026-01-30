from langchain_core.tools import tool
from agent_service.skills.rag import RAGSkill

# Default configuration for Huafei
DEFAULT_CONFIG = {
    "knowledge_collection": "huafei_knowledge_base",
    "style_collection": "huafei_style_corpus",
    "collection_name": "huafei_knowledge_base" # Fallback
}

@tool
async def retrieve_knowledge(query: str) -> str:
    """
    Retrieve Huafei's past knowledge, market analysis logic, and speaking style.
    Use this tool when you need to answer questions about market trends, valuation logic, specific sectors, or when you want to mimic his specific style of speaking.
    """
    skill = RAGSkill(config=DEFAULT_CONFIG)
    
    # Execute RAG
    # Input format for RAGSkill: {"query": "...", "top_k": 3}
    result = await skill.execute({"query": query, "top_k": 3})
    
    # Format the output for the LLM
    knowledge_list = result.get("knowledge", [])
    style_list = result.get("style", [])
    
    output = ""
    
    if knowledge_list:
        output += "### Relevant Knowledge & Market Views:\n"
        for i, item in enumerate(knowledge_list):
            content = item.get("content", str(item))
            output += f"{i+1}. {content}\n"
        output += "\n"
        
    if style_list:
        output += "### Style & Expression Examples:\n"
        for i, item in enumerate(style_list):
            content = item.get("content", str(item))
            output += f"{i+1}. {content}\n"
            
    if not output:
        return "No relevant knowledge or style examples found."
        
    return output
