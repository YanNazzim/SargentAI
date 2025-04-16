# SargentAI/app.py
import streamlit as st
import requests
import json
import re
import os
import socket # Added for server connection check

# --- LangChain Imports ---
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# -------------------------

# --- Streamlit App Setup ---
st.set_page_config(page_title="Sargent Manufacturing Support Chat") # Changed title
st.title("Sargent Manufacturing Support Chat")

# --- Configuration ---
MODEL_SERVER_URL_OLLAMA = "http://127.0.0.1:11434" # Base URL for ChatOllama
LOADED_MODEL_ID = "deepseek-r1:8b" # Your chat model (ensure it's pulled in Ollama)

PERSIST_DIRECTORY = "./chroma_db" # Same path used in ingest.py
EMBEDDING_MODEL_ID = "nomic-embed-text" # Same embedding model used in ingest.py
# --- End Configuration ---

# --- Helper function to check server connection ---
def check_server_connection(host, port):
    """Check if the Ollama server is reachable."""
    try:
        # Attempt to create a connection to the server host/port
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        # If connection fails, return False
        return False

# --- Initialize RAG Components (Load Vector Store & Set up Retriever) ---
@st.cache_resource # Cache resource so it doesn't reload on every interaction
def initialize_rag_components():
    # Check if the vector store directory exists
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"❌ Vector store not found at `{PERSIST_DIRECTORY}`. Please run the ingestion script (`ingest.py`).")
        return None, None # Return None if DB doesn't exist

    # Check if the Ollama server is running and reachable
    if not check_server_connection("localhost", 11434):
        st.error("❌ Could not connect to the Ollama server at http://localhost:11434. Please make sure `ollama serve` is running.")
        return None, None # Return None if server isn't reachable

    try:
        print("Initializing RAG components...")
        # Initialize embeddings using the specified model
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_ID)
        # Load the Chroma vector store from the persisted directory
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        # Create a retriever from the vector store to fetch relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={'k': 5}) # Retrieve top 5 relevant chunks
        print("RAG components initialized.")
        return retriever, embeddings
    except Exception as e:
        # Handle potential errors during initialization
        st.error(f"❌ Failed to initialize embeddings or vector store: {e}")
        return None, None

# --- Initialize Chat Model ---
@st.cache_resource
def initialize_llm():
    # Check server connection before initializing LLM
    if not check_server_connection("localhost", 11434):
        st.error("❌ Could not connect to the Ollama server at http://localhost:11434. Please make sure `ollama serve` is running.")
        return None
    try:
        print(f"Initializing LLM: {LOADED_MODEL_ID}")
        # Initialize the ChatOllama model
        llm = ChatOllama(model=LOADED_MODEL_ID, base_url=MODEL_SERVER_URL_OLLAMA)
        print("LLM initialized.")
        return llm
    except Exception as e:
        # Handle potential errors during LLM initialization
        st.error(f"❌ Failed to initialize LLM: {e}")
        return None

# --- Load RAG components and LLM ---
# Check if the vector store directory exists before attempting initialization
if os.path.exists(PERSIST_DIRECTORY):
    retriever, embeddings = initialize_rag_components()
    llm = initialize_llm()
    # RAG is enabled only if both retriever and LLM initialized successfully
    rag_enabled = retriever is not None and llm is not None
else:
    rag_enabled = False
    st.warning(f"❌ Knowledge base not found ({PERSIST_DIRECTORY}). Running without document search. Please run the ingestion script (`ingest.py`).")
    retriever = None
    llm = None # Ensure LLM is None if RAG components failed or DB is missing

# --- System Prompt for RAG (Modified) ---
RAG_SYSTEM_PROMPT_TEMPLATE = """You are a Tech Support Representative for Sargent Manufacturing. Your primary goal is to answer user questions *strictly based on the provided context* about Sargent door hardware products. Do not infer, assume, or add any information not explicitly present in the context.

When answering:
1.  **Base Answers Solely on Context:** Your entire response must be derived *directly* from the information given in the 'Context' section below.
2.  **Be Direct and Concise:** Directly address the user's specific question first.
3.  **Provide Explicit Details:** If the context contains relevant details (like features, options, specifications, notes, or documentation links) that directly answer the question, present them clearly. Only include details *explicitly mentioned* in the context.
4.  **Use Structure:** Use bullet points for lists (e.g., available functions, finishes, options, documentation links found in the context) for clarity and conciseness.
5.  **State Limitations:** If the provided context does *not* contain the information needed to answer the question, clearly state that the information is unavailable in the provided materials. Do not attempt to find the answer elsewhere or guess.
6.  **Handle Documentation Requests:** When asked for documentation (templates, manuals, instructions, etc.), look for relevant links (URLs) or file paths within the 'Context', especially in structured data fields like 'link', 'link1', 'text', 'text1'. List any found links/paths using bullet points. If none are found in the context, state that.

when user mentiones 80 series they mean all devices under wide80 series in the exitdevice data

**Crucially: Do not make assumptions, infer meanings, or provide information (even if generally known about Sargent products) that is not explicitly written in the context provided below.**

Context:
{context}
"""


# --- LangChain RAG Chain Setup ---
# Setup the RAG chain only if components are successfully loaded
if rag_enabled:
    # Create a prompt template that includes the system message, context, and chat history
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"), # Add chat history placeholder
        ("human", "{input}") # User's latest question/input
    ])

    # Chain to combine documents into the context
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Chain that retrieves documents first, then passes them and the input to the document_chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
else:
    # Ensure retrieval_chain is None if RAG isn't enabled
    retrieval_chain = None

# --- Session State Initialization ---
# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you with Sargent Manufacturing products today?"}
    ]

# --- Display Chat History ---
# Iterate through the stored messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Function to Clean Response ---
def remove_think_tags(text):
    """Removes <think>...</think> blocks from the LLM response."""
    if not isinstance(text, str):
        return text
    # Use regex to find and remove <think> blocks and any surrounding whitespace
    cleaned_text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

# --- Function to Get LLM Response (using LangChain RAG Chain) ---
def get_rag_response(user_input, chat_history):
    """Gets response from the RAG chain using user input and chat history."""
    # Check if RAG is properly set up
    if not rag_enabled or retrieval_chain is None:
        return "Error: RAG components are not initialized. Cannot search documents."

    # Convert Streamlit chat history to LangChain format if needed
    # (Often LangChain chains can handle list of dicts directly, but adjust if necessary)
    # formatted_history = [HumanMessage(content=m['content']) if m['role'] == 'user'
    #                      else AIMessage(content=m['content'])
    #                      for m in chat_history]

    try:
        print(f"\n--- Invoking RAG Chain ---")
        print(f"Input: {user_input}")
        # Invoke the retrieval chain with user input and chat history
        response = retrieval_chain.invoke({
            "input": user_input,
            "chat_history": chat_history # Pass the history directly
        })

        # Extract the answer from the response dictionary
        answer = response.get("answer", "Error: Could not get answer from RAG chain.")
        print(f"Raw Answer: {answer}")

        # Clean the final answer (e.g., remove specific tags if the model uses them)
        cleaned_answer = remove_think_tags(answer)
        print(f"Cleaned Answer: {cleaned_answer}")
        return cleaned_answer

    except Exception as e:
        # Handle errors during RAG chain invocation
        st.error(f"Error invoking RAG chain: {e}")
        # Provide a more informative error message if connection fails
        error_message = f"Error processing request: {e}. Is the Ollama server running and accessible at {MODEL_SERVER_URL_OLLAMA}?"
        print(error_message) # Log detailed error
        return "Sorry, I encountered an error trying to process your request. Please ensure the Ollama server is running."

# --- Handle User Input ---
# Get user input from the chat interface
if prompt := st.chat_input("Ask your question..."):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show spinner while generating response
    with st.spinner("Sargent AI is thinking..."):
        # Prepare chat history for the RAG chain (exclude the current user prompt)
        history_for_rag = st.session_state.messages[:-1]

        # Get the response using the RAG chain if enabled
        if rag_enabled:
            assistant_response = get_rag_response(prompt, history_for_rag)
        else:
            # Provide a fallback message if RAG is not enabled
            assistant_response = "Knowledge base not available. Cannot answer based on documents. Please run the ingest script."

    # Add assistant response to history and display it
    if assistant_response is not None:
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
    else:
        # Handle cases where response might be None (e.g., due to an error caught earlier)
        # A general error message is usually displayed by the function causing the error.
        pass