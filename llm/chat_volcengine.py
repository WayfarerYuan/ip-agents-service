from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.messages import AIMessageChunk

class ChatVolcengine(ChatOpenAI):
    """
    Custom ChatOpenAI subclass for Volcengine Ark API.
    Handles 'reasoning_content' field in streaming responses.
    """
    
    def _convert_chunk_to_generation_chunk(
        self, chunk: Any, default_class: Any = ChatGenerationChunk, *args, **kwargs
    ) -> ChatGenerationChunk:
        """
        Override to extract 'reasoning_content' from the raw chunk.
        """
        # Call super to get the basic chunk
        gen_chunk = super()._convert_chunk_to_generation_chunk(chunk, default_class, *args, **kwargs)
        
        try:
            # Try to find the delta
            delta = None
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
            elif isinstance(chunk, dict):
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
            
            if delta:
                reasoning = None
                
                # Check various locations for reasoning_content
                if hasattr(delta, "reasoning_content"):
                     reasoning = delta.reasoning_content
                elif isinstance(delta, dict) and "reasoning_content" in delta:
                     reasoning = delta["reasoning_content"]
                elif hasattr(delta, "model_extra") and delta.model_extra:
                     reasoning = delta.model_extra.get("reasoning_content")
                
                # Last resort: model_dump
                if not reasoning and hasattr(delta, "model_dump"):
                    d = delta.model_dump()
                    if "reasoning_content" in d:
                        reasoning = d["reasoning_content"]
                elif not reasoning and hasattr(delta, "dict"):
                    d = delta.dict()
                    if "reasoning_content" in d:
                        reasoning = d["reasoning_content"]

                if reasoning:
                    # Inject into additional_kwargs of the message
                    if not gen_chunk.message.additional_kwargs:
                        gen_chunk.message.additional_kwargs = {}
                    gen_chunk.message.additional_kwargs["reasoning_content"] = reasoning

        except Exception as e:
            # Silently fail to avoid disrupting the stream
            pass
            
        return gen_chunk
