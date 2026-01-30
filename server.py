import json
import sys
import os

# Ensure the parent directory is in sys.path so 'agent_service' can be imported as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage

from agent_service.graph import create_graph
from agent_service.database import AsyncMySQLSaver
from agent_service.utils import load_agent_config, get_db_connection

# Global State
graph_app = None
checkpointer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global graph_app, checkpointer
    async with AsyncMySQLSaver.from_conn_info() as saver:
        checkpointer = saver
        graph_app = create_graph(checkpointer)
        print("Agent Service initialized.")
        yield
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    agent_id: str
    message: str
    session_id: str
    user_id: str = "anonymous"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    agent_conf = load_agent_config(request.agent_id)
    if not agent_conf:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "agent_id": request.agent_id,
        "user_id": request.user_id,
        "agent_config": agent_conf["info"],
        "intent": {},
        "rag_data": {},
        "skill_results": {}
    }
    
    config = {"configurable": {"thread_id": request.session_id}}

    async def event_generator():
        # Using astream_events (v2) for granular control over streaming tokens and tool events.
        # version="v2" is required for standard event format.
        try:
            async for event in graph_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                
                # 1. Chat Token Streaming
                # Capture tokens from the 'chat_model' (ChatOpenAI)
                if kind == "on_chat_model_start":
                    # Capture the input messages to the LLM
                    inputs_data = event["data"].get("input")
                    print(f"[DEBUG] on_chat_model_start input: {inputs_data}") # Add debug print
                    
                    messages_list = []
                    # Check if input is directly a list (common in some invokes) or a dict with messages
                    if isinstance(inputs_data, list):
                        raw_msgs = inputs_data
                    elif isinstance(inputs_data, dict) and "messages" in inputs_data:
                        raw_msgs = inputs_data["messages"]
                    else:
                        # Fallback: try to serialize whatever it is
                        raw_msgs = []
                        
                    for m in raw_msgs:
                         # Handle list of lists if nested
                         if isinstance(m, list):
                             for sub_m in m:
                                 messages_list.append(sub_m.dict() if hasattr(sub_m, "dict") else str(sub_m))
                         else:
                             messages_list.append(m.dict() if hasattr(m, "dict") else str(m))
                    
                    yield f"data: {json.dumps({'type': 'llm_debug', 'data': messages_list})}\n\n"

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    # If it's a content chunk (not a tool call chunk)
                    if chunk.content:
                        # Clean up content if it's not string (rare in streaming but safe)
                        content = chunk.content
                        if isinstance(content, str):
                            yield f"data: {json.dumps({'type': 'message', 'content': content})}\n\n"
                            
                # 2. Tool Execution Debugging (Input)
                # Capture when a tool starts execution
                elif kind == "on_tool_start":
                    # We only care about our defined tools
                    name = event["name"]
                    # Skip internal LangChain tools if any (usually none here)
                    if name in ["web_search", "retrieve_knowledge", "search_ximalaya"]:
                        inputs_data = event["data"].get("input")
                        yield f"data: {json.dumps({'type': 'tool_debug', 'status': 'start', 'tool': name, 'input': inputs_data})}\n\n"
                        
                # 3. Tool Execution Debugging (Output) & Result Display
                # Capture when a tool finishes
                elif kind == "on_tool_end":
                    name = event["name"]
                    if name in ["web_search", "retrieve_knowledge", "search_ximalaya"]:
                        output_data = event["data"].get("output")
                        
                        # Send Debug Info
                        # Output might be long, but for debug panel we send it.
                        # Note: output_data is usually a string (Tool output)
                        if hasattr(output_data, "content"): # If it's a Message
                            output_str = output_data.content
                        else:
                            output_str = str(output_data)
                            
                        yield f"data: {json.dumps({'type': 'tool_debug', 'status': 'end', 'tool': name, 'output': output_str})}\n\n"
                        
                        # Send Legacy 'tool' event for compatibility (if frontend relies on it)
                        # The frontend likely uses 'rag_data' etc. from previous logic.
                        # We map tool names to data keys.
                        tool_key_map = {
                            "retrieve_knowledge": "rag_data",
                            "web_search": "web_search",
                            "search_ximalaya": "ximalaya_result"
                        }
                        
                        data_key = tool_key_map.get(name, name)
                        
                        # For RAG, frontend might expect nested content
                        if name == "retrieve_knowledge":
                             yield f"data: {json.dumps({'type': 'tool', 'data': {data_key: {'content': output_str}}})}\n\n"
                        else:
                             yield f"data: {json.dumps({'type': 'tool', 'data': {data_key: output_str}})}\n\n"

        except Exception as e:
            print(f"Stream Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/agents/{agent_id}")
def get_agent_info(agent_id: str):
    conf = load_agent_config(agent_id)
    if not conf:
        raise HTTPException(status_code=404, detail="Agent not found")
    info = conf["info"].copy()
    info["skills"] = list(conf["skills_map"].keys())
    return info

class PromptSettings(BaseModel):
    agent_id: str
    prompt_template: str = None
    # chat_prompt and intent_prompt are deprecated/unused
    # chat_prompt: str = None 
    # intent_prompt: str = None

@app.get("/settings/prompt")
def get_prompt_settings(agent_id: str):
    conf = load_agent_config(agent_id)
    if not conf:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    info = conf["info"]
    print(f"[DEBUG] get_prompt_settings info: {info}")
    return {
        "prompt_template": info.get("prompt_main")
    }

@app.post("/settings/prompt")
def save_prompt_settings(settings: PromptSettings):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Update agents table
            sql = "UPDATE agents SET "
            params = []
            updates = []
            
            if settings.prompt_template is not None:
                updates.append("prompt_main = %s")
                params.append(settings.prompt_template)
            
            if not updates:
                return {"status": "no changes"}
                
            sql += ", ".join(updates) + " WHERE agent_id = %s"
            params.append(settings.agent_id)
            
            cursor.execute(sql, tuple(params))
            conn.commit()
            
            # Clear cache to ensure next load gets fresh data
            from agent_service.utils import agent_cache
            if settings.agent_id in agent_cache:
                del agent_cache[settings.agent_id]
                
            return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
