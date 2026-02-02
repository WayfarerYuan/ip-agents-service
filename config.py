import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# MySQL 配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "10.146.2.144")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 13306))
MYSQL_USER = os.getenv("MYSQL_USER", "test")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ip_agents_db")

# Zilliz 配置 (Milvus)
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

# LLM 配置 (Doubao / Volcengine)
# 使用 Doubao-Seed-1.8
VOLC_API_KEY = os.getenv("VOLC_API_KEY")
VOLC_BASE_URL = os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_GENERATOR = "doubao-seed-1-8-251228" # Doubao Seed 1.8 Endpoint ID


# Web Search API (Volcengine)
VOLC_SEARCH_API_KEY = os.getenv("VOLC_SEARCH_API_KEY")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_EMBEDDING_NAME = "qwen/qwen3-embedding-8b"
