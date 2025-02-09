from typing import Protocol

from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    SystemMessage,
    UserMessage,
)


class IAzureDeepseek(Protocol):
    async def generate(
        self, prompts: list[SystemMessage | AssistantMessage | UserMessage], **kwargs
    ) -> ChatCompletions: ...
