"""
Streamlit UI for the AI PC Builder application.
This module provides a web-based interface for users to interact with the PC Builder agent,
enabling them to search for and receive recommendations on desktop computer components.

Features:
- Allows users to configure the system prompt for the agent
- Initializes the backend API with a custom or default prompt
- Provides a chat interface for users to describe their PC build requirements
- Displays agent responses and recommendations for computer components
- Maintains chat history for user and agent interactions

Functions:
    setup_api(prompt: str | None) -> bool:
        Initializes the backend model with a system prompt via API
    query_api(prompt: str) -> Optional[str]:
        Sends a user query to the backend API and returns the agent's response
    init_session_state():
        Initializes Streamlit session state variables
    main():
        Main entry point for the Streamlit UI application

Constants:
    API_URL: URL of the FastAPI backend
    TIMEOUT: HTTP request timeout configuration
    DEFAULT_SETUP_PROMPT: Default system prompt for agent initialization
"""

# Disable Streamlit usage statistics
from streamlit.web import bootstrap
bootstrap.load_config_options(flag_options={"browser.gatherUsageStats": False})

from typing import Optional, Literal
from pydantic import BaseModel
import httpx
import streamlit as st

# API Configuration
API_URL = "http://localhost:8000"
TIMEOUT = httpx.Timeout(600, connect=5, pool=5)


class ChatMessage(BaseModel):
    """Message model for chat interactions."""
    role: Literal["user", "assistant"]
    content: str


DEFAULT_SETUP_PROMPT = """You are an assistent whose goal is to help the user find where to buy desktop computer parts. You are to find these parts using a sequence of steps as below:
1. Inquire the user about the user country, city, budget and component type.
2. Use the search tools to find the retailers in the provided country and city.
3. Search retailer web site to find the best components in terms of price and performance that match user criteria.
4.Output the list of ten best components, for each component list the name, price, retailer name, retailer product id and timestamp when the informaion was obtained.
The returned data about components can not be older than 7 days before the current date.
"""

def setup_api(prompt: str | None) -> bool:
    """
    Setup the model via API endpoint with a system prompt.
    
    Args:
        prompt (str | None): System prompt for the model setup
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if prompt is None:
        prompt = DEFAULT_SETUP_PROMPT
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{API_URL}/setup",
                content=prompt,
                headers={"Content-Type": "text/plain"},
                timeout=TIMEOUT
            )
            return response.status_code == 200
    except httpx.RequestError:
        return False

def query_api(prompt: str) -> Optional[str]:
    """
    Send a query to the API and get response.
    
    Args:
        prompt (str): User's query text
        
    Returns:
        Optional[str]: Response from the API or None if request failed
    """
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{API_URL}/query",
                content=prompt,
                headers={"Content-Type": "text/plain"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return response_data.get("response", {}).get("content")
                except (KeyError, ValueError):
                    return None
            return None
    except httpx.RequestError:
        return None

def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "setup_prompt" not in st.session_state:
        st.session_state.setup_prompt = DEFAULT_SETUP_PROMPT
    if "api_ready" not in st.session_state:
        st.session_state.api_ready = False


def main():
    """Main Streamlit UI application."""
    st.title("🖥️ AI PC Builder")
    st.write("I'll help you find the perfect computer components for your needs!")

    # Initialize session state
    init_session_state()

    # Setup section
    with st.expander("⚙️ Setup", expanded=not st.session_state.api_ready):
        setup_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.setup_prompt,
            height=300,
            key="setup_prompt"
        )
        
        if st.button("� Initialize API"):
            with st.spinner("Setting up the model..."):
                if setup_api(setup_prompt):
                    st.session_state.api_ready = True
                    st.success("✅ API initialized successfully!")
                else:
                    st.error("❌ Failed to initialize API. Please check if the server is running.")

    if not st.session_state.api_ready:
        st.warning("⚠️ Please initialize the API before starting.")
        st.stop()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    # Chat input
    if prompt := st.chat_input("What kind of PC are you looking to build?"):
        # Add user message to chat history
        user_message = ChatMessage(role="user", content=prompt)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response through API
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if response := query_api(prompt):
                    st.markdown(response)
                    assistant_message = ChatMessage(role="assistant", content=response)
                    st.session_state.messages.append(assistant_message)
                else:
                    st.error("❌ Failed to get response from API. Please try again.")


if __name__ == "__main__":
    main()
