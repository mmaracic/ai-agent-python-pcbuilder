
import pytest
import os
from dotenv import load_dotenv
from embedding.azure_llm_embedder import AzureLlmEmbedder


@pytest.fixture
def env_vars():
    """
    Fixture to get environment variables.
    """
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    return {
        "endpoint": os.getenv("AZURE_EMBEDDER_ENDPOINT"),
        "api_version": os.getenv("AZURE_EMBEDDER_API_VERSION"),
        "deployment": os.getenv("AZURE_EMBEDDER_DEPLOYMENT"),
        "model": os.getenv("AZURE_EMBEDDER_MODEL", "text-embedding-3-large"),
        "api_key": os.getenv("AZURE_EMBEDDER_API_KEY"),
    }


# This is a separate fixture that creates a mock AzureOpenAI client
@pytest.fixture
def mock_azure_openai_for_embedding(monkeypatch):
    """Fixture to create a mock AzureOpenAI client that returns a predefined embedding"""
    mock_embedding = [0.1, 0.2, 0.3]
    
    class MockResponse:
        @property
        def data(self):
            return [type('obj', (object,), {'embedding': mock_embedding})]
    
    class MockAzureOpenAI:
        def __init__(self, azure_deployment, api_version, azure_endpoint, api_key):
            self.embeddings = self.MockEmbeddings()
        
        class MockEmbeddings:
            def create(self, input, model):
                return MockResponse()
    
    # Apply the mock
    monkeypatch.setattr("embedding.azure_llm_embedder.AzureOpenAI", MockAzureOpenAI)
    return mock_embedding


# This is a separate fixture that creates a mock AzureOpenAI client that returns an empty list
@pytest.fixture
def mock_azure_openai_for_empty(monkeypatch):
    """Fixture to create a mock AzureOpenAI client that returns an empty list"""
    class MockResponse:
        @property
        def data(self):
            return []
    
    class MockAzureOpenAI:
        def __init__(self, azure_deployment, api_version, azure_endpoint, api_key):
            self.embeddings = self.MockEmbeddings()
        
        class MockEmbeddings:
            def create(self, input, model):
                return MockResponse()
    
    # Apply the mock
    monkeypatch.setattr("embedding.azure_llm_embedder.AzureOpenAI", MockAzureOpenAI)


def test_embed_returns_embedding(mock_azure_openai_for_embedding, env_vars):
    """Test that the embed method returns the expected embedding."""
    # Create embedder after mock is applied
    embedder = AzureLlmEmbedder(
        endpoint=env_vars["endpoint"],
        api_version=env_vars["api_version"],
        deployment=env_vars["deployment"],
        model=env_vars["model"],
        api_key=env_vars["api_key"]
    )
    
    # Use the mocked embedder
    text = "test text"
    result = embedder.embed(text)
    
    # Assert using the mock_embedding value returned from the fixture
    assert result is not None and isinstance(result, list)
    assert len(result) > 0, "Embedding vector should not be empty"
    assert result == mock_azure_openai_for_embedding, "Embedding vector should match the mock data"


def test_embed_returns_empty_on_no_data(mock_azure_openai_for_empty, env_vars):
    """Test that the embed method returns an empty list when no data is returned."""
    # Create embedder after mock is applied
    embedder = AzureLlmEmbedder(
        endpoint=env_vars["endpoint"],
        api_version=env_vars["api_version"],
        deployment=env_vars["deployment"],
        model=env_vars["model"],
        api_key=env_vars["api_key"]
    )
    
    # Use the mocked embedder
    text = "test text"
    result = embedder.embed(text)
    
    assert result == []


# For the real test, we use a fixture to create the real embedder
@pytest.fixture
def real_embedder(env_vars):
    """Fixture to create a real AzureLlmEmbedder"""
    return AzureLlmEmbedder(
        endpoint=env_vars["endpoint"],
        api_version=env_vars["api_version"],
        deployment=env_vars["deployment"],
        model=env_vars["model"],
        api_key=env_vars["api_key"]
    )


def test_real_embedding(real_embedder, env_vars):
    """
    Integration test that calls the actual Azure OpenAI service.
    Requires actual credentials to be set in the environment.
    """
    # Skip if credentials are not available
    if not all([env_vars["endpoint"], env_vars["api_version"], 
                env_vars["deployment"], env_vars["model"], env_vars["api_key"]]):
        pytest.skip("Azure embedding credentials not available in environment")
        
    # Use the real embedder
    text = "This is a test sentence for embedding."
    result = real_embedder.embed(text)
    
    # Verify that we got a reasonable embedding vector back
    assert isinstance(result, list)
    assert len(result) > 0, "Embedding vector should not be empty"
    assert isinstance(result[0], float), "Embedding vector should contain float values"
    
    # Most embedding models produce normalized vectors
    vector_magnitude = sum(val**2 for val in result)**0.5
    assert abs(vector_magnitude - 1.0) < 0.1, "Embedding vector should be approximately normalized"
