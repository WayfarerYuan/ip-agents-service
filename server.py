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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
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
    # Standard Xiaoya Protocol Fields
    uid: str = Field(description="用户ID", default="anonymous")
    query: str = Field(description="用户提问")
    agentScene: str = Field(description="场景ID (Agent ID)")
    deviceId: str = Field(description="设备ID", default="unknown_device")
    
    # Optional Fields
    requestId: str = Field(default=None)
    conversationId: str = Field(default=None)
    clientInfo: dict = Field(default_factory=dict)
    contextInfo: dict = Field(default_factory=dict)
    customInputs: dict = Field(default_factory=dict)

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, background_tasks: BackgroundTasks):
    # 1. Adapt Input Parameters
    user_id = request.uid
    message = request.query
    agent_id = request.agentScene
    
    # Session/Thread ID
    session_id = request.conversationId or request.requestId or f"sess_{user_id}_{int(datetime.now().timestamp())}"

    agent_conf = load_agent_config(agent_id)
    if not agent_conf:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
        
    # 2. Fetch Core Memory (Runtime Context)
    core_memory = ""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT core_memory FROM user_memories WHERE user_id = %s AND agent_id = %s",
                (user_id, agent_id)
            )
            res = cursor.fetchone()
            if res:
                core_memory = res.get('core_memory', "")
    finally:
        conn.close()
    
    # Configure Model and Extra Body from customInputs
    # customInputs structure: { "model_config": { "model": "...", "enable_thinking": true, "reasoning_effort": "medium" } }
    # Or flat: { "model": "...", ... }
    # Let's support flat for simplicity in first pass, or check specifically
    
    custom_inputs = request.customInputs or {}
    model_name = custom_inputs.get("model") or MODEL_GENERATOR
    enable_thinking = custom_inputs.get("enable_thinking")
    reasoning_effort = custom_inputs.get("reasoning_effort")
    
    extra_body = {}
    
    # Deep Thinking Logic
    if enable_thinking is not None:
        extra_body["thinking"] = {
            "type": "enabled" if enable_thinking else "disabled"
        }
    
    if reasoning_effort:
        if "reasoning" not in extra_body:
            extra_body["reasoning"] = {}
        extra_body["reasoning"]["effort"] = reasoning_effort

    agent_conf["info"]["model_override"] = model_name
    agent_conf["info"]["extra_body"] = extra_body

    inputs = {
        "messages": [HumanMessage(content=message)],
        "agent_id": agent_id,
        "user_id": user_id,
        "agent_config": agent_conf["info"],
        "intent": {},
        "rag_data": {},
        "skill_results": {},
    }
    
    inputs["agent_config"]["core_memory_context"] = core_memory
    
    config = {"configurable": {"thread_id": session_id}}
    
    interaction_log = [] 

    async def event_generator():
        interaction_log.append(HumanMessage(content=message))
        ai_response_content = ""
        
        try:
            async for event in graph_app.astream_events(inputs, config=config, version="v2"):
                kind = event["event"]
                
                # 1. Chat Token Streaming
                if kind == "on_chat_model_start":
                    pass # Debug logic omitted

                elif kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    
                    # 1.1 Capture Thinking
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                    if reasoning:
                        # Use 'iting' protocol for Thinking Process
                        payload = {
                            "type": "iting",
                            "data": {
                                "type": 40,
                                "value": {
                                    "cardType": "thinking",
                                    "cardData": {
                                        "delta": reasoning,
                                        "state": "processing"
                                    }
                                }
                            }
                        }
                        yield f"event: iting\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    # 1.2 Capture Content
                    if chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            ai_response_content += content 
                            payload = {
                                "type": "writtenAnswer",
                                "data": {
                                    "delta": content,
                                    "end": False,
                                    "final": False
                                }
                            }
                            yield f"event: writtenAnswer\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                            
                # 2. Tool Execution
                elif kind == "on_tool_start":
                    name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    
                    # Emit 'iting' event for Tool Call
                    payload = {
                        "type": "iting",
                        "data": {
                            "type": 40,
                            "value": {
                                "cardType": "tool_call",
                                "cardData": {
                                    "tool": name,
                                    "status": "start",
                                    "input": tool_input
                                }
                            }
                        }
                    }
                    yield f"event: iting\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    
                    # Also emit debug event for Debug Panel
                    debug_payload = {
                        "type": "tool_debug",
                        "data": {
                            "tool": name,
                            "status": "start",
                            "input": tool_input
                        }
                    }
                    yield f"event: debug\ndata: {json.dumps(debug_payload, ensure_ascii=False)}\n\n"
                        
                elif kind == "on_tool_end":
                    name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    output_str = str(output)
                    
                    # Emit 'iting' event for Tool Result
                    payload = {
                        "type": "iting",
                        "data": {
                            "type": 40,
                            "value": {
                                "cardType": "tool_call",
                                "cardData": {
                                    "tool": name,
                                    "status": "end",
                                    "output": output_str[:500] if output_str else ""
                                }
                            }
                        }
                    }
                    yield f"event: iting\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    # Also emit debug event for Debug Panel
                    debug_payload = {
                        "type": "tool_debug",
                        "data": {
                            "tool": name,
                            "status": "end",
                            "output": output_str[:500] if output_str else ""
                        }
                    }
                    yield f"event: debug\ndata: {json.dumps(debug_payload, ensure_ascii=False)}\n\n"
                    
                    if name == "display_card":
                        try:
                            # Parse inner card data
                            if isinstance(event["data"].get("output"), dict):
                                card_inner = event["data"].get("output")
                            else:
                                card_inner = json.loads(output_str)
                                
                            # Wrap in Xiaoya 'iting' protocol
                            payload = {
                                "type": "iting",
                                "data": {
                                    "type": 40,
                                    "value": card_inner
                                }
                            }
                            yield f"event: iting\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                        except Exception as e:
                            print(f"Error parsing display_card output: {e}")

        except Exception as e:
            print(f"Stream Error: {e}")
            yield f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n"
            
        # End event (Xiaoya protocol)
        yield f"event: end\ndata: {json.dumps({'type': 'end', 'data': {'success': True, 'msg': ''}}, ensure_ascii=False)}\n\n"
        
        # 3. Generate Reply Suggestions (Post-processing)
        try:
            # Call LLM to generate suggestions
            if ai_response_content and VOLC_API_KEY:
                suggestion_prompt = f"""
                Based on the conversation history and the last AI response, generate 3 short, relevant reply suggestions for the user.
                
                [Last AI Response]
                {ai_response_content}
                
                [Output Format]
                Return ONLY a JSON object with a "suggestions" key containing a list of 3 strings.
                Example: {{"suggestions": ["Tell me more", "Why?", "Next step"]}}
                """
                
                llm = ChatOpenAI(
                    api_key=VOLC_API_KEY,
                    base_url=VOLC_BASE_URL,
                    model=MODEL_GENERATOR,
                    temperature=0.7
                )
                
                # Use a separate invocation to get suggestions
                # We do this after the main stream is done (but before connection closes? actually connection might close)
                # Wait, 'yield' keeps connection open.
                
                # Note: This is a synchronous call in an async generator, might block slightly. 
                # Ideally use ainvoke but we are in async generator.
                
                suggestion_response = await llm.ainvoke([HumanMessage(content=suggestion_prompt)])
                content = suggestion_response.content
                
                # Extract JSON
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    suggestions_data = json.loads(json_str)
                    
                    # Emit 'iting' event for Reply Suggestions
                    payload = {
                        "type": "iting",
                        "data": {
                            "type": 40,
                            "value": {
                                "cardType": "reply_suggestions",
                                "cardData": suggestions_data
                            }
                        }
                    }
                    yield f"event: iting\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    
        except Exception as e:
            print(f"Error generating suggestions: {e}")

        if ai_response_content:
            interaction_log.append(AIMessage(content=ai_response_content))
            background_tasks.add_task(update_core_memory, user_id, agent_id, interaction_log)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/agents")
def get_all_agents():
    """Get list of all available agents"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT agent_id, name, description, avatar_url FROM agents ORDER BY created_at DESC")
            agents = cursor.fetchall()
            return agents
    finally:
        conn.close()

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
def get_skills(agent_id: str = Query(..., description="Agent ID")):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT skill_id, name, description FROM skills WHERE agent_id = %s ORDER BY created_at DESC", 
                (agent_id,)
            )
            skills = cursor.fetchall()
            return skills
    finally:
        conn.close()

@app.get("/skills/{skill_id}")
def get_skill_details(skill_id: str, agent_id: str = Query(..., description="Agent ID")):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM skills WHERE skill_id = %s AND agent_id = %s", (skill_id, agent_id))
            skill = cursor.fetchone()
            if not skill:
                raise HTTPException(status_code=404, detail="Skill not found")
            return skill
    finally:
        conn.close()

@app.post("/skills")
def create_or_update_skill(skill: Skill, agent_id: str = Query(..., description="Agent ID")):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Upsert (新结构：skills 表包含 agent_id)
            sql = """
                INSERT INTO skills (skill_id, agent_id, name, description, content, config, enabled) 
                VALUES (%s, %s, %s, %s, %s, %s, TRUE)
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                description = VALUES(description),
                content = VALUES(content),
                config = VALUES(config)
            """
            cursor.execute(sql, (skill.skill_id, agent_id, skill.name, skill.description, skill.content, skill.config_schema))
            conn.commit()
            return {"status": "success", "skill_id": skill.skill_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/skills/{skill_id}")
def delete_skill(skill_id: str, agent_id: str = Query(..., description="Agent ID")):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check if skill exists and belongs to this agent
            cursor.execute("SELECT skill_id FROM skills WHERE skill_id = %s AND agent_id = %s", (skill_id, agent_id))
            skill = cursor.fetchone()
            if not skill:
                raise HTTPException(status_code=404, detail="Skill not found or not authorized")
                
            cursor.execute("DELETE FROM skills WHERE skill_id = %s AND agent_id = %s", (skill_id, agent_id))
            conn.commit()
            return {"status": "success"}
    finally:
        conn.close()

@app.get("/models")
def get_models():
    """Get list of available LLM models"""
    return [
        # 1. Models with Reasoning Effort (Low/Medium/High)
        {
            "id": "doubao-seed-1-8-251228",
            "name": "Doubao Seed 1.8 (Thinking)",
            "has_thinking": True,
            "thinking_mode": "effort"
        },
        {
            "id": "doubao-seed-1-6-251015",
            "name": "Doubao Seed 1.6 (Thinking)",
            "has_thinking": True,
            "thinking_mode": "effort"
        },
        {
            "id": "doubao-seed-1-6-lite-251015",
            "name": "Doubao Seed 1.6 Lite (Thinking)",
            "has_thinking": True,
            "thinking_mode": "effort"
        },
        
        # 2. Models with Thinking Toggle Only
        {
            "id": "glm-4-7-251222",
            "name": "GLM 4.7 (Thinking)",
            "has_thinking": True,
            "thinking_mode": "toggle"
        },
        {
            "id": "deepseek-v3-2-251201",
            "name": "DeepSeek V3 (Thinking)",
            "has_thinking": True,
            "thinking_mode": "toggle"
        },
        {
            "id": "kimi-k2-thinking-251104",
            "name": "Kimi K2 (Thinking)",
            "has_thinking": True,
            "thinking_mode": "toggle"
        },

        # 3. Models without Thinking Mode
        {
            "id": "doubao-1-5-pro-32k-250115",
            "name": "Doubao 1.5 Pro",
            "has_thinking": False
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=18002)
