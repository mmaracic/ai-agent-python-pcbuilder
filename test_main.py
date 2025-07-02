import os
import pytest
from fastapi.testclient import TestClient
from main import app, OPEN_ROUTER_API_KEY, MODEL_NOT_INITIALIZED_ERROR

client = TestClient(app)

@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Set a dummy API key for testing."""
    monkeypatch.setenv(OPEN_ROUTER_API_KEY, "test-api-key")

def test_setup_success():
    response = client.post(
        "/setup",
        data="You are a helpful assistant.",
        headers={"Content-Type": "text/plain"}
    )
    # FastAPI returns 200 if a response is returned, 204 if None is returned
    assert response.status_code in (200, 204)

def test_setup_missing_api_key(monkeypatch):
    """Test that missing API key raises an exception when calling /setup."""
    monkeypatch.delenv(OPEN_ROUTER_API_KEY, raising=False)
    with pytest.raises(Exception) as exc_info:
        client.post(
            "/setup",
            data="Prompt",
            headers={"Content-Type": "text/plain"}
        )
    assert "OPEN_ROUTER_API_KEY environment variable is not set" in str(exc_info.value)

def test_query_without_setup():
    # Reset app state to simulate not initialized
    app.state.app_state.compiled_graph = None
    # Ensure model and prompt_template are also None to trigger the error
    app.state.app_state.model = None
    app.state.app_state.prompt_template = None
    response = client.post(
        "/query",
        data="Hello",
        headers={"Content-Type": "text/plain"}
    )
    # If /query endpoint returns error message in JSON, check for it
    assert response.status_code in (200, 400, 500)
    # Accept either key, depending on how error is returned
    assert (
        MODEL_NOT_INITIALIZED_ERROR in response.text
        or "Model is not initialized" in response.text
    )