from abc import ABC, abstractmethod

class BaseGuardrail(ABC):
    @abstractmethod
    async def check(self, text: str) -> bool:
        """Return True if the text passes this guardrail, False if it should be blocked"""