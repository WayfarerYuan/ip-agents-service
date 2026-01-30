from langchain_core.tools import tool
from agent_service.utils import get_db_connection

@tool
def load_skill(skill_name: str) -> str:
    """
    Load a specialized skill or Standard Operating Procedure (SOP) by name.
    Use this tool when you need specific instructions, domain knowledge, or a checklist to complete a complex task.
    
    Args:
        skill_name: The name or keyword of the skill to load (e.g., "deep_research", "stock_analysis").
    
    Returns:
        The content (SOP/Instructions) of the skill if found, or a "not found" message.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Fuzzy search by name or exact match by ID
            query = """
                SELECT name, content, description 
                FROM skills 
                WHERE skill_id = %s OR name LIKE %s
                LIMIT 1
            """
            cursor.execute(query, (skill_name, f"%{skill_name}%"))
            skill = cursor.fetchone()
            
            if not skill:
                # Fallback: List available skills to help the agent correct itself
                cursor.execute("SELECT name, description FROM skills LIMIT 10")
                available = cursor.fetchall()
                available_list = "\n".join([f"- {s['name']}: {s['description']}" for s in available])
                return f"Skill '{skill_name}' not found. Available skills:\n{available_list}"
            
            content = skill.get('content')
            if not content:
                return f"Skill '{skill['name']}' found but has no content/SOP defined."
                
            return f"### Loaded Skill: {skill['name']}\n\n{content}"
    except Exception as e:
        return f"Error loading skill: {str(e)}"
    finally:
        conn.close()
