import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# MySQL 配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "10.146.2.144")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 13306))
MYSQL_USER = os.getenv("MYSQL_USER", "test")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "QnKL1yscJKH9p7XXeB")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "ip_agents_db")

# Zilliz 配置 (Milvus)
MILVUS_URI = os.getenv("MILVUS_URI", "https://in05-45e89c08b93bfd5.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "a5cac67e41944fb39f959b3c4a380bb36623e4b03ccd0f85bbbebdc540737016aa116d6e360e2ca88fa4597723226ecee36f88aa")

# LLM 配置 (Doubao / Volcengine)
# 使用 Doubao-Seed-1.8
VOLC_API_KEY = os.getenv("VOLC_API_KEY", "70eb18d7-9e2c-4ba6-acaa-c5f420873b9f")
VOLC_BASE_URL = os.getenv("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_GENERATOR = "doubao-seed-1-8-251228" # Doubao Seed 1.8 Endpoint ID
MODEL_EMBEDDING = "doubao-embedding-v2" 

# Web Search API (Volcengine)
VOLC_SEARCH_API_KEY = os.getenv("VOLC_SEARCH_API_KEY", "StA5Uy1gue2l6gZlMLlx4vDRviRUbo0j")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_EMBEDDING_NAME = "qwen/qwen3-embedding-8b"
