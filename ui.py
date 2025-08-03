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
- Enables searching the database via API for components using text queries

Functions:
    setup_api(prompt: str | None) -> bool:
        Initializes the backend model with a system prompt via API
    query_api(prompt: str) -> list[ChatMessage]:
        Sends a user query to the backend API and returns the agent's response messages
    query_db_api(text: str) -> list[RetrievedDatabaseExtractedItem]:
        Retrieves items from the database via API using text similarity search
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

import os
from typing import Optional, Literal, List
from pydantic import BaseModel
from database.retrieved_item_model import RetrievedDatabaseExtractedItem
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
    name: Optional[str] = None  # For tool messages, includes the tool name


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
    st.session_state.pc_builder_ready = False
        
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

def query_db_api(text: str, max_results: int = 10) -> list[RetrievedDatabaseExtractedItem]:
    """
    Query the backend API's database using text embedding similarity search.
    
    Args:
        text (str): User's query text to be embedded and used for similarity search
        max_results (int, optional): Maximum number of results to return. Defaults to 10.
        
    Returns:
        list[RetrievedDatabaseExtractedItem]: List of retrieved items
        
    Raises:
        ValueError: If API connection fails or the server returns an error
        Exception: For unexpected errors
    """
    try:
        # Create the request payload as JSON
        payload = {
            "text": text,
            "max_results": max_results,
            "user_id": "default_user"
        }
        
        with httpx.Client() as client:
            response = client.post(
                f"{API_URL}/query_db",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                try:
                    items = response.json()
                    # Convert to RetrievedDatabaseExtractedItem objects
                    retrieved_items = [
                        RetrievedDatabaseExtractedItem(**item)
                        for item in items
                    ]
                    return retrieved_items
                except (KeyError, ValueError) as e:
                    raise ValueError(f"Failed to parse API response: {str(e)}")
            else:
                raise ValueError(f"API returned error status code: {response.status_code}")
    except httpx.RequestError as e:
        raise ValueError(f"Failed to connect to API: {str(e)}")
        




def init_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "setup_prompt" not in st.session_state:
        st.session_state.setup_prompt = DEFAULT_SETUP_PROMPT
    if "pc_builder_ready" not in st.session_state:
        st.session_state.pc_builder_ready = False
    if "db_query_history" not in st.session_state:
        st.session_state.db_query_history = []
    if "retrieved_items" not in st.session_state:
        st.session_state.retrieved_items = []
    if "previous_query" not in st.session_state:
        st.session_state.previous_query = ""


def main():
    """Main Streamlit UI application."""
    # Set page layout to use more screen width
    st.set_page_config(
        page_title="AI PC Builder",
        layout="wide",  # Use the wide layout mode
        initial_sidebar_state="collapsed"
    )
    
    # Create a centered container for our content
    container = st.container()
    with container:
        # Add some padding on the sides to avoid using the full width
        _, center_col, _ = st.columns([1, 8, 1])
        
        with center_col:
            st.title("🖥️ AI PC Builder")
            st.markdown("*Your intelligent assistant for computer components*")
            
            # Initialize session state
            init_session_state()
            
            # Create tabs
            tab1, tab2 = st.tabs(["PC Builder", "Database Query"])
    
            # Tab 1: PC Builder (original UI)
            with tab1:
                st.markdown("### Find the perfect computer components for your needs")
                st.write("Chat with our AI assistant to discover the best PC parts available at local retailers.")
                
                # Setup section (moved back to first tab)
                with st.expander("⚙️ Setup", expanded=not st.session_state.pc_builder_ready):
                    setup_prompt = st.text_area(
                        "System Prompt",
                        value=st.session_state.setup_prompt,
                        height=300,
                        key="setup_prompt"
                    )
            
                    if st.button("Initialize PC Builder API"):
                        with st.spinner("Setting up the model..."):
                            if setup_api(setup_prompt):
                                st.success("✅ PC Builder API initialized successfully!")
                                st.session_state.pc_builder_ready = True
                            else:
                                st.error("❌ Failed to initialize API. Please check if the server is running.")
                    if not st.session_state.pc_builder_ready:
                        st.warning("⚠️ Please initialize the PC Builder API before starting.")
                        # Display warning but allow tabs to remain visible
                
                if st.session_state.pc_builder_ready:
                    # Create a container for the chat interface
                    chat_container = st.container()
                    
                    # Display chat history in the container
                    for message in st.session_state.messages:
                        style = MESSAGE_STYLES.get(message.role, "")
                        with st.chat_message(message.role):
                            st.markdown(f'<div style="{style}">{message.content}</div>', unsafe_allow_html=True)

                    if ((len(st.session_state.messages)>0 and st.session_state.messages[-1].role == "assistant") or
                        len(st.session_state.messages) == 0):
                        # Process the input if the user has entered something
                        prompt = st.chat_input("What kind of PC are you looking to build?", key="pc_builder_input")

                        if prompt:
                            # Show user message in chat
                            st.session_state.messages.append(ChatMessage(role="user", content=prompt))
                            st.rerun()  # Rerun to update chat display
                    else:
                        prompt = st.session_state.messages[-1].content
                        # Get agent response through API
                        with st.spinner("🤔 Thinking..."):
                            messages = query_api(prompt)
                            if messages:
                                st.session_state.messages.extend(messages)
                                st.rerun()  # Rerun to update chat display
                            else:
                                st.error("❌ Failed to get response from API. Please try again.")
            
            # Tab 2: Database Query
            with tab2:
                st.markdown("### Database Component Search")
                st.write("Search for components in the database using natural language queries and semantic similarity.")
                
                # Query input - always enabled, errors will be shown only when search is attempted
                col1, col2 = st.columns([5, 1])
                with col1:
                    query = st.text_input(
                        "Enter your search query:",
                        placeholder="E.g., 'gaming GPU under $500'",
                        key="db_query_input",
                        on_change=None  # This allows the on_submit callback to work
                    )
                with col2:
                    search_button = st.button(
                        "🔍 Search",
                        use_container_width=True
                    )
            
                # Results count slider
                max_results = st.slider(
                    "Maximum number of results:",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    help="Select how many results to retrieve at maximum"
                )
                
                # Process database query when search button is clicked or Enter is pressed
                is_new_query = query != st.session_state.get("previous_query", "")
                if (search_button or is_new_query) and query:
                    # Update previous query in session state
                    st.session_state.previous_query = query
                    
                    try:
                        with st.spinner("Searching database..."):
                            # Add query to history
                            if query not in st.session_state.db_query_history:
                                st.session_state.db_query_history.append(query)
                            
                            # Get items from database via API
                            retrieved_items = query_db_api(query, max_results)
                            st.session_state.retrieved_items = retrieved_items
                            
                            if retrieved_items:
                                st.success(f"Found {len(retrieved_items)} items")
                            else:
                                st.warning("No items found matching your query")
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                        st.error("Please check if the API server is running and properly configured.")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                        st.error("Please check application logs for details.")
        
                # Display search history
                if st.session_state.db_query_history:
                    with st.expander("Search History", expanded=False):
                        for idx, past_query in enumerate(st.session_state.db_query_history):
                            if st.button(f"Rerun: {past_query}", key=f"history_{idx}"):
                                try:
                                    with st.spinner("Searching database..."):
                                        # Get items from database via API using current max_results setting
                                        retrieved_items = query_db_api(past_query, max_results)
                                        st.session_state.retrieved_items = retrieved_items
                                        
                                        if retrieved_items:
                                            st.success(f"Found {len(retrieved_items)} items")
                                        else:
                                            st.warning("No items found matching your query")
                                except ValueError as e:
                                    st.error(f"❌ {str(e)}")
                                    st.error("Please check if the API server is running and properly configured.")
                                except Exception as e:
                                    st.error(f"❌ Unexpected error: {str(e)}")
                                    st.error("Please check application logs for details.")
        
                # Display retrieved items
                if st.session_state.retrieved_items:
                    st.markdown("### 📊 Search Results")
                    
                    # Create a table for the items with similarity as the first column
                    table_data = [
                        {
                            "Similarity": f"{item.similarity_score:.4f}",
                            "Description": item.description,
                            "Price": item.price,
                            "Store": item.store_name,
                            "Item Code": item.item_code,
                            "Date Retrieved": item.date_time
                        }
                        for item in st.session_state.retrieved_items
                    ]
                    
                    # Make the table wider by setting the container width and using st.container()
                    with st.container():
                        st.dataframe(
                            table_data, 
                            use_container_width=True,
                            column_config={
                                "Similarity": st.column_config.NumberColumn(
                                    "Similarity",
                                    help="Semantic similarity score",
                                    format="%.4f",
                                    width="small"
                                ),
                                "Description": st.column_config.TextColumn(
                                    "Description",
                                    width="large"
                                ),
                                "Price": st.column_config.TextColumn(
                                    "Price",
                                    width="small"
                                ),
                                "Store": st.column_config.TextColumn(
                                    "Store",
                                    width="medium"
                                ),
                                "Item Code": st.column_config.TextColumn(
                                    "Item Code",
                                    width="small"
                                ),
                                "Date Retrieved": st.column_config.DatetimeColumn(
                                    "Date Retrieved",
                                    format="D MMM YYYY, h:mm a",
                                    width="medium"
                                )
                            },
                            height=400
                        )
            
                    # Allow downloading results as CSV
                    csv_data = "\n".join([
                        "Similarity,Description,Price,Store,Item Code,Date Retrieved",
                        *[f"{item['Similarity']},\"{item['Description']}\",\"{item['Price']}\",\"{item['Store']}\",\"{item['Item Code']}\",\"{item['Date Retrieved']}\"" 
                          for item in table_data]
                    ])
                    
                    if st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="search_results.csv",
                        mime="text/csv",
                    ):
                        st.success("Download started!")


if __name__ == "__main__":
    main()
