from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from agent_service.state import AgentState
from agent_service.nodes.chat import chat_worker
from agent_service.nodes.setup import setup_worker
from agent_service.tools.web_search import web_search
from agent_service.tools.rag import retrieve_knowledge
from agent_service.tools.ximalaya import search_ximalaya

def create_graph(checkpointer=None):
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("setup", setup_worker)
    workflow.add_node("chat", chat_worker)
    
    # ToolNode with all available tools
    tools = [web_search, retrieve_knowledge, search_ximalaya]
    workflow.add_node("tools", ToolNode(tools))

    # Entry
    workflow.set_entry_point("setup")

    # Edges
    workflow.add_edge("setup", "chat")
    
    # Chat -> Tools (if tool_calls) OR END
    # If tools_condition returns "tools", it goes to "tools" node.
    # If it returns END (no tool calls), the graph finishes.
    workflow.add_conditional_edges(
        "chat",
        tools_condition,
    )
    
    # Tools -> Chat (loop back to generate response using tool outputs)
    workflow.add_edge("tools", "chat")

    return workflow.compile(checkpointer=checkpointer)
