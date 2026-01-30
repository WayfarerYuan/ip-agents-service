import asyncio
import pymysql
from agent_service.config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

def init_db():
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD
    )
    cursor = conn.cursor()
    
    # Create Database
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute(f"USE {MYSQL_DATABASE}")
    
    # Table: Agents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        agent_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        avatar_url VARCHAR(500),
        system_prompt TEXT,
        collection_name VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table: Skills
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS skills (
        skill_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100) NOT NULL,
        description TEXT,
        config_schema JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table: Agent_Skills (Many-to-Many)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_skills (
        id INT AUTO_INCREMENT PRIMARY KEY,
        agent_id VARCHAR(50),
        skill_id VARCHAR(50),
        config JSON,
        enabled BOOLEAN DEFAULT TRUE,
        FOREIGN KEY (agent_id) REFERENCES agents(agent_id),
        FOREIGN KEY (skill_id) REFERENCES skills(skill_id),
        UNIQUE KEY unique_agent_skill (agent_id, skill_id)
    )
    """)
    
    # Table: User_Settings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_settings (
        user_id VARCHAR(100),
        agent_id VARCHAR(50),
        settings JSON,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, agent_id)
    )
    """)
    
    # Table: Chat_Logs
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id VARCHAR(100),
        agent_id VARCHAR(50),
        user_message TEXT,
        assistant_reply TEXT,
        metadata JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Table: Checkpoints (for LangGraph persistence)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS checkpoints (
        thread_id VARCHAR(255) NOT NULL,
        checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR(255) NOT NULL,
        parent_checkpoint_id VARCHAR(255),
        type VARCHAR(50),
        checkpoint LONGBLOB,
        metadata LONGBLOB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS checkpoint_writes (
        thread_id VARCHAR(255) NOT NULL,
        checkpoint_ns VARCHAR(255) NOT NULL DEFAULT '',
        checkpoint_id VARCHAR(255) NOT NULL,
        task_id VARCHAR(255) NOT NULL,
        idx INT NOT NULL,
        channel VARCHAR(255) NOT NULL,
        type VARCHAR(50),
        value LONGBLOB,
        PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    )
    """)
    
    # Seed Data: Skills
    skills = [
        ('ximalaya_skill', '喜马拉雅音频能力', '查询主播专辑和声音', '{"anchor_id": "int"}'),
        ('stock_analysis_skill', '股市行情分析', '查询大盘和个股分析', '{}'),
        ('rag_skill', '知识库检索', '检索Zilliz向量库', '{"collection_name": "string"}')
    ]
    
    for s_id, s_name, s_desc, s_conf in skills:
        cursor.execute(
            "INSERT IGNORE INTO skills (skill_id, name, description, config_schema) VALUES (%s, %s, %s, %s)",
            (s_id, s_name, s_desc, s_conf)
        )
        
    # Seed Data: Agent (Huafei)
    huafei_prompt = """你就是**华飞**。
你是一位专业的财经主播，也是一位深谙人性的生活观察者。
说话风格：专业、犀利、透彻，喜欢用数据说话，但也不乏人文关怀。
"""
    cursor.execute(
        "INSERT IGNORE INTO agents (agent_id, name, description, avatar_url, system_prompt, collection_name) VALUES (%s, %s, %s, %s, %s, %s)",
        ('huafei', '华飞', '财经主播，股市分析专家', '', huafei_prompt, 'huafei_knowledge_base')
    )
    
    # Seed Data: Agent Skills Bindings
    # 1. Ximalaya (Anchor ID: 41855169)
    cursor.execute(
        "INSERT IGNORE INTO agent_skills (agent_id, skill_id, config) VALUES (%s, %s, %s)",
        ('huafei', 'ximalaya_skill', '{"anchor_id": 41855169}')
    )
    # 2. Stock Analysis
    cursor.execute(
        "INSERT IGNORE INTO agent_skills (agent_id, skill_id, config) VALUES (%s, %s, %s)",
        ('huafei', 'stock_analysis_skill', '{}')
    )
    # 3. RAG
    cursor.execute(
        "INSERT IGNORE INTO agent_skills (agent_id, skill_id, config) VALUES (%s, %s, %s)",
        ('huafei', 'rag_skill', '{"collection_name": "huafei_knowledge_base"}')
    )
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
