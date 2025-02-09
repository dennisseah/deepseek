import asyncio

from azure.ai.inference.models import SystemMessage

from az_deepseek.hosting import container
from az_deepseek.protocols.i_azure_deepseek import IAzureDeepseek

svc = container.resolve(IAzureDeepseek)
asyncio.run(
    svc.generate(
        [
            SystemMessage(
                content="Can help me understand why we have to pay attention to the climate change issues?"  # noqa E501
            )
        ],
    )
)
