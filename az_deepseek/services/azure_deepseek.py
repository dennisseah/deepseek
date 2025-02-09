from dataclasses import dataclass

from azure.ai.inference.aio import ChatCompletionsClient
from azure.ai.inference.models import (
    AssistantMessage,
    ChatCompletions,
    SystemMessage,
    UserMessage,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from lagom.environment import Env

from az_deepseek.protocols.i_azure_deepseek import IAzureDeepseek


class AzureDeepseekEnv(Env):
    azure_deepseek_endpoint: str
    azure_deepseek_key: str | None = None


@dataclass
class AzureDeepseek(IAzureDeepseek):
    env: AzureDeepseekEnv

    def get_client(self):
        if self.env.azure_deepseek_key:
            return ChatCompletionsClient(
                endpoint=self.env.azure_deepseek_endpoint,
                credential=AzureKeyCredential(self.env.azure_deepseek_key),
            )

        return ChatCompletionsClient(
            endpoint=self.env.azure_deepseek_endpoint,
            credential=DefaultAzureCredential(),  # type: ignore
        )

    async def generate(
        self, prompts: list[SystemMessage | AssistantMessage | UserMessage], **kwargs
    ) -> ChatCompletions:
        client = self.get_client()
        try:
            result: ChatCompletions = await client.complete(messages=prompts, **kwargs)  # type: ignore
            return result
        finally:
            await client.close()
