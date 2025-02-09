"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from az_deepseek.protocols.i_azure_deepseek import IAzureDeepseek

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def _() -> logging.Logger:
    return logging.getLogger("az_deepseek")


@dependency_definition(container, singleton=True)
def azure_llama_service() -> IAzureDeepseek:
    from az_deepseek.services.azure_deepseek import AzureDeepseek

    return container[AzureDeepseek]
