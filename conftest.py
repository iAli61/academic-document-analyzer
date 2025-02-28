import pytest
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

@pytest.fixture(scope="session")
def ml_client():
    client = MLClient.from_config(
        credential=DefaultAzureCredential()
    )
    yield client

@pytest.fixture(scope="session")
def setup_environment():
    # Add any necessary setup for the Azure ML environment or dependencies here
    pass