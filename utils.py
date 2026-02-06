import json
import pymysql
from agent_service.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

agent_cache = {}

def get_db_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        cursorclass=pymysql.cursors.DictCursor
    )

def load_agent_config(agent_id: str):
    if agent_id in agent_cache:
        return agent_cache[agent_id]
        
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Load Agent Basic Info
            cursor.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
            agent = cursor.fetchone()
            print(f"[DEBUG] DB fetched agent: {agent}") # Added debug log
            if not agent:
                return None
            
            # Load Skills (新结构：skills 表包含 agent_id)
            cursor.execute("""
                SELECT skill_id, name, config 
                FROM skills
                WHERE agent_id = %s AND enabled = TRUE
            """, (agent_id,))
            skills_data = cursor.fetchall()
            
            # Instantiate Skills
            skills_map = {}
            for s in skills_data:
                config = json.loads(s['config']) if isinstance(s['config'], str) else s['config']
                # Just store metadata, do not instantiate classes
                skills_map[s['skill_id']] = {
                    "name": s['name'],
                    "config": config
                }
                
            agent_config = {
                "info": agent,
                "skills_map": skills_map
            }
            agent_cache[agent_id] = agent_config
            return agent_config
    finally:
        conn.close()
