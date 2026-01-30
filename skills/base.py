from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSkill(ABC):
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the skill"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what the skill does"""
        pass

    @abstractmethod
    async def execute(self, input_data: Any) -> Any:
        """Execute the skill logic"""
        pass
