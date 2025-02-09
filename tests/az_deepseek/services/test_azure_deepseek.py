from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from azure.ai.inference.models import ChatCompletions, CompletionsUsage, SystemMessage
from pytest_mock import MockerFixture

from az_deepseek.services.azure_deepseek import (
    AzureDeepseek,
    AzureDeepseekEnv,
)


@pytest.fixture
def mock_client():
    mock_client = AsyncMock()
    mock_client.complete.return_value = ChatCompletions(
        id="fake",
        created=datetime.now(),
        model="fake",
        usage=CompletionsUsage(completion_tokens=10, prompt_tokens=10, total_tokens=20),
        choices=[],
    )
    return mock_client


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [None, "fake"])
async def test_generate(mocker: MockerFixture, api_key, mock_client):
    mocker.patch(
        "az_deepseek.services.azure_deepseek.ChatCompletionsClient",
        return_value=mock_client,
    )

    service = AzureDeepseek(
        AzureDeepseekEnv(
            azure_deepseek_endpoint="https://api.openai.com",
            azure_deepseek_key=api_key,
        )
    )

    result = await service.generate(
        [SystemMessage(content="Hello there")],
    )

    assert result is not None
