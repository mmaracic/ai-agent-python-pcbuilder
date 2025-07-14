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

# CSS styles for message types
MESSAGE_STYLES = {
    "user": """
        background-color: #1E88E5;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    """,
    "assistant": """
        background-color: #D32F2F;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    """,
    "tool": """
        background-color: #2E7D32;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    """
}


class ChatMessage(BaseModel):
    """Message model for chat interactions."""
    role: Literal["user", "assistant", "tool"]
    content: str
    name: Optional[str] = None  # For tool messages, can include tool name


DEFAULT_SETUP_PROMPT = """
You are an assistent whose goal is to help the user find at which local retailer to buy desktop computer parts.
The user has to specify city and country where to search for retailers.
Think about the steps how to fulfill the user request.
The final result of the plan has to be the list of ten best components across all available retailers that fit user request.
For each component list the name, price, retailer name, retailer product id and timestamp when the informaion was obtained.
The returned data about components can not be older than 7 days before the current date.
If search parameters are expanded all the retailers have to be searched again.
"""

def setup_api(prompt: str | None) -> bool:
    """
    Setup the model via API endpoint with a system prompt and clear session state.
    
    Args:
        prompt (str | None): System prompt for the model setup
    
    Returns:
        bool: True if setup was successful, False otherwise
    """
    if prompt is None:
        prompt = DEFAULT_SETUP_PROMPT

    # Clear session state
    st.session_state.messages = []
    st.session_state.api_ready = False
        
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

def query_api(prompt: str) -> list[ChatMessage]:
    """
    Send a query to the API and get response messages.
    
    Args:
        prompt (str): User's query text
        
    Returns:
        list[ChatMessage]: List of response messages or empty list if request failed
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
                    messages = response_data.get("response", [])
                    chat_messages = []
                    
                    for msg in messages:
                        # Map message types to roles
                        role = "assistant" if msg.get("type") == "ai" else "tool"
                        content = msg.get("content", "")
                        name = msg.get("name", None)  # For tool messages, can include tool name
                        if content:  # Only add messages with content
                            chat_messages.append(ChatMessage(role=role, content=content, name=name))

                    return chat_messages
                except (KeyError, ValueError):
                    return []
            return []
    except httpx.RequestError:
        return []

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
        style = MESSAGE_STYLES.get(message.role, "")
        with st.chat_message(message.role):
            st.markdown(f'<div style="{style}">{message.content}</div>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("What kind of PC are you looking to build?"):
        # Add user message to chat history
        user_message = ChatMessage(role="user", content=prompt)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(f'<div style="{MESSAGE_STYLES["user"]}">{prompt}</div>', unsafe_allow_html=True)

        # Get agent response through API
        with st.spinner("🤔 Thinking..."):
            messages = query_api(prompt)
            if messages:
                for message in messages:
                    style = MESSAGE_STYLES.get(message.role, "")
                    with st.chat_message(message.role):
                        if message.role == "tool":
                            st.markdown(f'<div style="{style}"><strong>Tool: {message.name}</strong> {message.content}</div>', unsafe_allow_html=True)
                        else:
                            # For user and assistant messages, just display the content
                            st.markdown(f'<div style="{style}">{message.content}</div>', unsafe_allow_html=True)
                    st.session_state.messages.append(message)
            else:
                st.error("❌ Failed to get response from API. Please try again.")


if __name__ == "__main__":
    main()
