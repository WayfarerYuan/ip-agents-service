from agent_service.state import AgentState
from agent_service.utils import load_agent_config

def setup_worker(state: AgentState):
    """
    Setup node to load configuration if missing.
    Useful for 'langgraph dev' where config is not injected by server.py.
    """
    # If agent_config is already present (e.g. from server.py), do nothing
    if state.get("agent_config"):
        return {}

    # If agent_id is provided, try to load config
    agent_id = state.get("agent_id")
    if agent_id:
        print(f"[setup] Loading config for agent_id: {agent_id}")
        config_data = load_agent_config(agent_id)
        if config_data and "info" in config_data:
            return {"agent_config": config_data["info"]}
    
    # Fallback or default if no agent_id
    # We could return a default config here if we wanted to be very helpful
    print("[setup] No agent_config or agent_id found. Using defaults.")
    return {}
