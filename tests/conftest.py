import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_qdrant_client(mocker):
    return mocker.patch("qdrant_client.QdrantClient")

@pytest.fixture
def mock_openai_client(mocker):
    return mocker.patch("openai.OpenAI")
