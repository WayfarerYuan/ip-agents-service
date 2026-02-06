from langchain_core.messages import SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from agent_service.llm.chat_volcengine import ChatVolcengine
from agent_service.state import AgentState
from agent_service.config import VOLC_API_KEY, VOLC_BASE_URL, MODEL_GENERATOR
from agent_service.tools.web_search import web_search
from agent_service.tools.rag import retrieve_knowledge
from agent_service.tools.ximalaya import search_ximalaya
from agent_service.tools.skills import load_skill
from agent_service.tools.cards import display_card

chat_model_default = ChatVolcengine(
    api_key=VOLC_API_KEY,
    base_url=VOLC_BASE_URL,
    model=MODEL_GENERATOR,
    temperature=0.7,
    streaming=True
)

async def chat_worker(state: AgentState):
    agent_config = state.get("agent_config", {})
    system_prompt_template = agent_config.get("prompt_main") or "You are a helpful assistant."
    
    # Dynamic Model Configuration
    model_name = agent_config.get("model_override", MODEL_GENERATOR)
    extra_body = agent_config.get("extra_body", None)
    
    if model_name != MODEL_GENERATOR or extra_body:
        print(f"[DEBUG] Using dynamic model: {model_name} with extra_body: {extra_body}")
        if not VOLC_API_KEY:
            raise ValueError("VOLC_API_KEY is not set. Please set it in the environment variables.")
        chat_model = ChatVolcengine(
            api_key=VOLC_API_KEY,
            base_url=VOLC_BASE_URL,
            model=model_name,
            temperature=0.7,
            streaming=True,
            model_kwargs={"extra_body": extra_body} if extra_body else {}
        )
    else:
        chat_model = chat_model_default
    
    # 1. Inject Core Memory (if exists)
    core_memory_text = agent_config.get("core_memory_context", "")
    if core_memory_text:
        system_prompt_template = f"【关于用户的核心记忆】\n{core_memory_text}\n\n" + system_prompt_template
    
    # Context is now provided via ToolMessages in the message history, 
    # so we don't need to manually inject rag_data or skill_results anymore.
    
    # Progressive Disclosure Instruction
    skill_instruction = """
    \n\n[Skill Loading Capabilities]
    You have access to a library of specialized skills and SOPs.
    If the user asks for a complex task (e.g., "analyze stocks", "write a report", "debug code"), 
    ALWAYS check if a relevant skill exists by using the `load_skill` tool first.
    Do not guess or hallucinate steps; load the official skill/SOP to ensure compliance.

    [Card Display Capabilities]
    You have access to a `display_card` tool to show structured UI elements.
    Use this tool when you need to display:
    - Questions with options (type='question')
    - Assessment feedback and next steps (type='assessment')
    - Final results or reports (type='result')
    - Subscription prompts (type='subscription')
    - Briefings (type='briefing')
    ALWAYS prefer using `display_card` over plain text for interactive flows.
    """
    
    final_system_prompt = system_prompt_template + skill_instruction
    
    # Ensure system prompt is the first message
    # We reconstruct the message list to update the system prompt if it changed
    # But we must preserve the conversation history.
    # Typically, state["messages"] contains the full history including the initial system prompt?
    # LangGraph adds messages. If we want to *replace* the system prompt, we might need to filter.
    # However, usually we just prepend the current system prompt to the *rest* of the history
    # and let the model handle it. 
    # Or, if state["messages"] accumulates, we might have multiple system prompts.
    # For now, let's assume we prepend a fresh system message.
    
    messages = [SystemMessage(content=final_system_prompt)] + state["messages"]
    
    # Bind ALL available tools
    # This enables "On-Demand" usage of RAG, Search, and other skills.
    tools = [web_search, retrieve_knowledge, search_ximalaya, load_skill, display_card]
    model_with_tools = chat_model.bind_tools(tools)
    
    # Debug: Print messages payload
    print(f"\n[DEBUG] Turn with {len(messages)} messages:")
    for i, m in enumerate(messages):
        content_preview = str(m.content)[:100] if m.content else "None"
        print(f"  [{i}] {m.type}: {content_preview}")
        if hasattr(m, "tool_calls") and m.tool_calls:
            print(f"      tool_calls: {m.tool_calls}")
    
    response = await model_with_tools.ainvoke(messages)
    
    # 1. Handle Tool Calls
    if response.tool_calls:
        print(f"[DEBUG] Model triggered tool calls: {response.tool_calls}")
        return {"messages": [response]}
    
    # 2. Handle Text Response (Cleanup Doubao list content if present)
    final_content = response.content
    if isinstance(response.content, list):
        print("[DEBUG] Detected list content, cleaning up...")
        text_parts = []
        for block in response.content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        final_content = "".join(text_parts)
    
    # Create a FRESH AIMessage to ensure clean history for next turn
    clean_response = AIMessage(content=final_content)
    print(f"[DEBUG] Returning clean AIMessage. Content len: {len(final_content)}")
        
    return {"messages": [clean_response]}
