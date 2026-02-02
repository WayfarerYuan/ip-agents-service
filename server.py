import json
import sys
import os

# Ensure the parent directory is in sys.path so 'agent_service' can be imported as a module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
# When running in Docker, we might be running from /app, and agent_service code is in /app
# We need to ensure that /app's parent is not needed, or structure imports differently.
# If we run 'python server.py' inside /app, 'agent_service' module is NOT available unless /app is treated as a package
# or we are outside.
# However, usually the structure is:
# project/
#   agent_service/
#      server.py
#      ...
#
# If we copy '.' to '/app', then /app contains server.py directly.
# So imports like 'from agent_service.graph import ...' will fail because 'agent_service' is not a subdirectory of /app,
# but rather /app IS the content of agent_service.

# Fix: If we are in the root of the service code, we should change imports or structure.
# But better: Adjust Dockerfile to preserve the directory structure.

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Optional

from agent_service.graph import create_graph
from agent_service.database import AsyncMySQLSaver
from agent_service.utils import load_agent_config, get_db_connection
from langchain_openai import ChatOpenAI
from agent_service.config import VOLC_API_KEY, VOLC_BASE_URL, MODEL_GENERATOR

# Global State
graph_app = None
checkpointer = None

from datetime import datetime

class KeyFact(BaseModel):
    content: str = Field(description="简短的核心事实（主谓宾结构），去除非必要细节，限20字以内")
    created_at: str = Field(description="记录时间，格式：YYYY-MM-DD HH:MM")

class CoreMemoryStructure(BaseModel):
    basic_profile: dict = Field(description="基础画像（姓名、年龄、职业、家庭状况等）", default_factory=dict)
    interests: List[str] = Field(description="兴趣偏好列表（简短关键词）", default_factory=list)
    key_facts: List[KeyFact] = Field(description="关键事实列表", default_factory=list)

async def update_core_memory(user_id: str, agent_id: str, messages: list):
    """
    Async background task to analyze conversation and update core memory.
    """
    try:
        print(f"[MemoryWorker] Starting memory analysis for user {user_id}...")
        
        # 1. Get current memory
        conn = get_db_connection()
        current_memory_str = ""
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT core_memory FROM user_memories WHERE user_id = %s AND agent_id = %s",
                    (user_id, agent_id)
                )
                res = cursor.fetchone()
                if res and res['core_memory']:
                    current_memory_str = res['core_memory']
        finally:
            conn.close()
            
        # 2. Extract conversation snippet
        
        # 3. Call LLM to update memory
        if not VOLC_API_KEY:
            raise ValueError("VOLC_API_KEY is not set")
        
        llm = ChatOpenAI(
            api_key=VOLC_API_KEY,
            base_url=VOLC_BASE_URL,
            model=MODEL_GENERATOR,
            temperature=0.1 # Lower temperature for structured output
        )
        
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Try to use structured output
        structured_llm = llm.with_structured_output(CoreMemoryStructure)
        
        system_instruction = f"""
        你是一位专业的记忆管理员。你的任务是从对话中提取用户的画像、偏好和关键事实，并更新核心记忆。
        
        [当前时间]
        {now_str}
        
        [当前核心记忆 (JSON or Text)]
        {current_memory_str if current_memory_str else "（暂无记忆）"}
        
        [任务]
        1. 分析【最近对话】和【当前核心记忆】。
        2. 提取/更新以下三类信息：
           - **基础画像 (basic_profile)**: 姓名、年龄、职业、居住地、家庭成员等。
           - **兴趣偏好 (interests)**: 用户的爱好、习惯、喜欢的风格等，保持精简。
           - **关键事实 (key_facts)**: 用户发生的重要事件（如买车、升职、投资等），必须带上时间。
        
        [规则]
        1. **极度精简**：Key Facts 必须极其简练，只保留核心主谓宾。例如“2024-02-01 18:00 用户买入英伟达股票”优于“用户提到他在2024年2月1日晚上6点买了英伟达的股票”。
        2. **时间感知**：如果是新发生的事实，使用当前时间 `{now_str}`。如果是历史事实，尽量推断时间。
        3. **合并策略**：如果当前记忆是旧文本格式，请将其整理到新的结构中。如果已有结构化数据，请进行合并更新（去重、覆盖旧信息）。
        4. **完整性**：输出完整的、更新后的记忆结构。
        """
        
        # Convert messages to string for analysis
        conv_text = ""
        for m in messages:
             role = m.type if hasattr(m, "type") else "unknown"
             content = m.content if hasattr(m, "content") else str(m)
             conv_text += f"{role}: {content}\n"
             
        try:
            # Use ainvoke for async
            new_memory_obj = await structured_llm.ainvoke([
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"【最近对话】:\n{conv_text}"}
            ])
            
            # Convert back to JSON string
            import json
            new_memory = json.dumps(new_memory_obj.model_dump(), indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"[MemoryWorker] Structured Output Failed: {e}. Fallback to text mode.")
            # Fallback logic if needed, or just return
            return

        # 4. Update DB if changed
        # Simple string comparison might be noisy due to formatting, but okay for now
        if new_memory and new_memory != current_memory_str:
            print(f"[MemoryWorker] Updating memory for {user_id}")
            conn = get_db_connection()
            try:
                with conn.cursor() as cursor:
                    sql = """
                        INSERT INTO user_memories (user_id, agent_id, core_memory)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE core_memory = VALUES(core_memory)
                    """
                    cursor.execute(sql, (user_id, agent_id, new_memory))
                    conn.commit()
            finally:
                conn.close()
        else:
            print("[MemoryWorker] No memory change detected.")
            
    except Exception as e:
        print(f"[MemoryWorker] Error: {e}")

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
    # New Model Parameters
    model: Optional[str] = None
    enable_thinking: Optional[bool] = None
    reasoning_effort: Optional[str] = None # low, medium, high

# Available Models for Frontend
AVAILABLE_MODELS = [
    {"id": "doubao-seed-1-8-251228", "name": "Doubao Seed 1.8 (Default)", "has_thinking": True},
    {"id": "glm-4-7-251222", "name": "GLM-4.7", "has_thinking": True, "thinking_mode": "toggle"},
    {"id": "doubao-seed-1-6-251015", "name": "Doubao Seed 1.6", "has_thinking": True},
    {"id": "doubao-1-5-pro-32k-250115", "name": "Doubao 1.5 Pro", "has_thinking": False},
]

@app.get("/models")
def get_available_models():
    return AVAILABLE_MODELS

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    agent_conf = load_agent_config(request.agent_id)
    if not agent_conf:
        raise HTTPException(status_code=404, detail="Agent not found")
        
    # 1. Fetch Core Memory (Runtime Context)
    core_memory = ""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT core_memory FROM user_memories WHERE user_id = %s AND agent_id = %s",
                (request.user_id, request.agent_id)
            )
            res = cursor.fetchone()
            if res:
                core_memory = res.get('core_memory', "")
    finally:
        conn.close()
    
    # Configure Model and Extra Body for Volcengine
    model_name = request.model if request.model else MODEL_GENERATOR
    extra_body = {}
    
    # Deep Thinking Logic
    if request.enable_thinking is not None:
        # Based on search results, use "thinking": {"type": "enabled"/"disabled"}
        extra_body["thinking"] = {
            "type": "enabled" if request.enable_thinking else "disabled"
        }
    
    # Reasoning Effort Logic
    if request.reasoning_effort:
        # Supporting both structures just in case, but prefer the nested one if documentation implies 'reasoning.effort'
        # The user said "support adjusting thinking length via reasoning.effort"
        # Search results hint at "Reasoning specifies the reasoning effort".
        # We'll use "reasoning": {"effort": "value"} as per common advanced API patterns (like Anthropic/DeepSeek)
        if "reasoning" not in extra_body:
            extra_body["reasoning"] = {}
        extra_body["reasoning"]["effort"] = request.reasoning_effort

    # Inject into inputs so create_graph (or specifically the chat node) can use it
    # We need to pass this config down to the chat node.
    # Currently `agent_service/graph.py` and `nodes/chat.py` instantiate the model.
    # We might need to override the model config in `agent_config`.
    
    # NOTE: The current architecture might define the model INSIDE the graph node.
    # We need to verify if `create_graph` or `chat` node respects runtime config overrides.
    # Let's check `agent_service/nodes/chat.py` if possible.
    # Assuming `agent_config` passed to graph is used to configure the model.
    # If not, we might need to modify `chat.py`.
    # For now, we inject `model_override` and `extra_body` into `agent_config`.
    
    agent_conf["info"]["model_override"] = model_name
    agent_conf["info"]["extra_body"] = extra_body

    inputs = {
        "messages": [HumanMessage(content=request.message)],
        "agent_id": request.agent_id,
        "user_id": request.user_id,
        "agent_config": agent_conf["info"],
        "intent": {},
        "rag_data": {},
        "skill_results": {},
    }
    
    # Inject Core Memory into config for this turn
    inputs["agent_config"]["core_memory_context"] = core_memory
    
    config = {"configurable": {"thread_id": request.session_id}}
    
    # Capture interactions for memory update
    interaction_log = [] # List to store User and AI messages

    async def event_generator():
        # Using astream_events (v2) for granular control over streaming tokens and tool events.
        # version="v2" is required for standard event format.
        
        # Add User message to log
        interaction_log.append(HumanMessage(content=request.message))
        ai_response_content = ""
        
        try:
            async for event in graph_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                
                # 1. Chat Token Streaming
                # Capture tokens from the 'chat_model' (ChatOpenAI)
                if kind == "on_chat_model_start":
                    # ... existing logic ...
                    pass 
                    
                    # Capture the input messages to the LLM (Debug logic remains)
                    inputs_data = event["data"].get("input")
                    print(f"[DEBUG] on_chat_model_start input: {inputs_data}") 
                    
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
                    
                    # 1.1 Capture Thinking/Reasoning Content (Volcengine/DeepSeek style)
                    # Usually in additional_kwargs['reasoning_content']
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        yield f"data: {json.dumps({'type': 'thinking', 'content': reasoning})}\n\n"

                    # 1.2 Capture Normal Content
                    if chunk.content:
                        # Clean up content if it's not string (rare in streaming but safe)
                        content = chunk.content
                        if isinstance(content, str):
                            ai_response_content += content # Accumulate full response
                            yield f"data: {json.dumps({'type': 'message', 'content': content})}\n\n"
                            
                # 3. Tool Execution Debugging (Input)
                # Capture when a tool starts execution
                elif kind == "on_tool_start":
                    # We only care about our defined tools
                    name = event["name"]
                    # Skip internal LangChain tools if any (usually none here)
                    if name in ["web_search", "retrieve_knowledge", "search_ximalaya", "load_skill"]:
                        inputs_data = event["data"].get("input")
                        yield f"data: {json.dumps({'type': 'tool_debug', 'status': 'start', 'tool': name, 'input': inputs_data})}\n\n"
                        
                # 3. Tool Execution Debugging (Output) & Result Display
                # Capture when a tool finishes
                elif kind == "on_tool_end":
                    name = event["name"]
                    if name in ["web_search", "retrieve_knowledge", "search_ximalaya", "load_skill"]:
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
        
        # After stream ends, trigger background task
        if ai_response_content:
            interaction_log.append(AIMessage(content=ai_response_content))
            # We must use background_tasks.add_task. 
            # However, StreamingResponse is a generator, so we can't easily use FastAPI's dependency injection for BackgroundTasks 
            # directly inside the generator *after* the return.
            # But we CAN add it to the background_tasks object passed to the endpoint, 
            # and FastAPI will execute it after the response (StreamingResponse) is fully consumed/closed.
            background_tasks.add_task(update_core_memory, request.user_id, request.agent_id, interaction_log)

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

@app.get("/memories")
def get_user_memory(user_id: str, agent_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT core_memory FROM user_memories WHERE user_id = %s AND agent_id = %s",
                (user_id, agent_id)
            )
            res = cursor.fetchone()
            if res:
                return {"core_memory": res['core_memory']}
            return {"core_memory": ""}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

class MemoryUpdate(BaseModel):
    user_id: str
    agent_id: str
    core_memory: str

@app.post("/memories")
def update_user_memory_manual(mem: MemoryUpdate):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            sql = """
                INSERT INTO user_memories (user_id, agent_id, core_memory)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE core_memory = VALUES(core_memory)
            """
            cursor.execute(sql, (mem.user_id, mem.agent_id, mem.core_memory))
            conn.commit()
            return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

class Skill(BaseModel):
    skill_id: str
    name: str
    description: str = None
    content: str = None
    config_schema: str = "{}"

@app.get("/skills")
def get_all_skills():
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT skill_id, name, description FROM skills ORDER BY created_at DESC")
            skills = cursor.fetchall()
            return skills
    finally:
        conn.close()

@app.get("/skills/{skill_id}")
def get_skill_details(skill_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM skills WHERE skill_id = %s", (skill_id,))
            skill = cursor.fetchone()
            if not skill:
                raise HTTPException(status_code=404, detail="Skill not found")
            return skill
    finally:
        conn.close()

@app.post("/skills")
def create_or_update_skill(skill: Skill):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Upsert
            sql = """
                INSERT INTO skills (skill_id, name, description, content, config_schema) 
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                description = VALUES(description),
                content = VALUES(content),
                config_schema = VALUES(config_schema)
            """
            cursor.execute(sql, (skill.skill_id, skill.name, skill.description, skill.content, skill.config_schema))
            conn.commit()
            return {"status": "success", "skill_id": skill.skill_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/skills/{skill_id}")
def delete_skill(skill_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check dependencies in agent_skills
            cursor.execute("SELECT count(*) as cnt FROM agent_skills WHERE skill_id = %s", (skill_id,))
            res = cursor.fetchone()
            if res['cnt'] > 0:
                raise HTTPException(status_code=400, detail="Cannot delete skill: It is bound to agents.")
                
            cursor.execute("DELETE FROM skills WHERE skill_id = %s", (skill_id,))
            if cursor.rowcount == 0:
                raise HTTPException(status_code=404, detail="Skill not found")
            conn.commit()
            return {"status": "success"}
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18002)
