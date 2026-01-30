from typing import Any, Dict, List
from pymilvus import MilvusClient
from agent_service.skills.base import BaseSkill
from agent_service.config import MILVUS_URI, MILVUS_TOKEN, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL_EMBEDDING_NAME
from openai import AsyncOpenAI

class RAGSkill(BaseSkill):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Support dual collections or single fallback
        self.collection_name = self.config.get("collection_name")
        self.knowledge_collection = self.config.get("knowledge_collection", self.collection_name)
        self.style_collection = self.config.get("style_collection", self.collection_name)
        
        self.milvus_client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
        self.embedding_client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL
        )

    @property
    def name(self) -> str:
        return "rag_skill"

    @property
    def description(self) -> str:
        return "Retrieve knowledge and style from vector database."

    async def _get_embedding(self, text: str) -> List[float]:
        text = text.strip() or "empty"
        response = await self.embedding_client.embeddings.create(
            model=MODEL_EMBEDDING_NAME,
            input=text
        )
        return response.data[0].embedding

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        Input: {"query": "用户问题", "top_k": 3}
        """
        query = input_data.get("query", "")
        top_k = input_data.get("top_k", 3)
        
        if not self.knowledge_collection and not self.style_collection:
            return {"error": "Collection names not configured"}

        vector = await self._get_embedding(query)
        
        knowledge = []
        style = []

        # Search for Knowledge
        if self.knowledge_collection:
            try:
                # If using separate collections, we might not need type filter, 
                # but keeping it loose (no filter) or strictly 'knowledge' depends on data ingestion.
                # Assuming separate collections contain purely their respective data types.
                # However, to be safe and compatible with mixed collections, we check if they are the same.
                
                filter_expr = "type == 'knowledge' or type == 'qa_pair'" if self.knowledge_collection == self.style_collection else ""
                
                res_know = self.milvus_client.search(
                    collection_name=self.knowledge_collection,
                    data=[vector],
                    limit=top_k,
                    filter=filter_expr,
                    output_fields=["content", "topic", "type"]
                )
                knowledge = [hit['entity'] for hit in res_know[0]] if res_know else []
            except Exception as e:
                print(f"[RAG] Knowledge search failed: {e}")
        
        # Search for Style
        if self.style_collection:
            try:
                filter_expr = "type == 'style'" if self.knowledge_collection == self.style_collection else ""
                
                res_style = self.milvus_client.search(
                    collection_name=self.style_collection, 
                    data=[vector],
                    limit=top_k,
                    filter=filter_expr,
                    output_fields=["content", "topic", "type"]
                )
                style = [hit['entity'] for hit in res_style[0]] if res_style else []
            except Exception as e:
                print(f"[RAG] Style search failed: {e}")
        
        return {
            "knowledge": knowledge,
            "style": style
        }
