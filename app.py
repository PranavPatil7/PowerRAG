import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
import requests


from llm_chat import Direct_LLM_Chat
from pdf_chat import PDF_QA_Session
from rag_chat import RAG_Chat
from rag_memory import Create_RAG_Memory


FAISS_INDEX_PATH = "faiss_index.bin"
RAG_DOCS_DIR = "rag_documents"

st.set_page_config(page_title="Aby's Assistant", layout="wide")
st.title("Aby's Assistant: LLM & RAG Modes")

# --- Initialize Session State ---
if "ollama_base_url" not in st.session_state:
    st.session_state.ollama_base_url = "http://localhost:11434"
if "selected_llm_model" not in st.session_state:
    st.session_state.selected_llm_model = None
if "selected_embedding_model" not in st.session_state:
    st.session_state.selected_embedding_model = "nomic-embed-text" 
if "rag_vector_store" not in st.session_state:
    st.session_state.rag_vector_store = None
if "pdf_vector_store" not in st.session_state: 
    st.session_state.pdf_vector_store = None
if "pdf_processed_for_qa" not in st.session_state:
    st.session_state.pdf_processed_for_qa = False
if "rag_memory_loaded" not in st.session_state: 
    st.session_state.rag_memory_loaded = False

os.makedirs(RAG_DOCS_DIR, exist_ok=True)

# --- Utility Functions ---
@st.cache_data(show_spinner=False)
def get_ollama_models(base_url):
    """Fetches a list of available Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        models = [m['name'] for m in models_data.get('models', [])]
        return models
    except requests.exceptions.ConnectionError:
        st.error(f"Error: Could not connect to Ollama server at {base_url}. "
                 "Please ensure Ollama is running (`ollama serve`).")
        return []
    except requests.exceptions.Timeout:
        st.error(f"Error: Ollama server at {base_url} timed out. It might be busy or unreachable.")
        return []
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Ollama models: {e}")
        return []


st.sidebar.markdown("---")
st.sidebar.markdown("### Global Settings")

available_models = get_ollama_models(st.session_state.ollama_base_url)

if available_models:
    try:
        default_llm_index = available_models.index(st.session_state.selected_llm_model) if st.session_state.selected_llm_model in available_models else 0
    except ValueError:
        default_llm_index = 0
    st.session_state.selected_llm_model = st.sidebar.selectbox(
        "Select LLM (Question Answering) Model",
        available_models,
        index=default_llm_index,
        key="llm_model_select"
    )

    # Embedding Model Selection (CRITICAL: Needs to be a dedicated embedding model like nomic-embed-text)
    embedding_models_options = [m for m in available_models if "embed" in m.lower() or "nomic" in m.lower() or "bge" in m.lower()]
    if not embedding_models_options:
        embedding_models_options = available_models 

    try:
        default_embed_index = embedding_models_options.index(st.session_state.selected_embedding_model) if st.session_state.selected_embedding_model in embedding_models_options else 0
    except ValueError:
        default_embed_index = 0
    st.session_state.selected_embedding_model = st.sidebar.selectbox(
        "Select Embedding Model (for PDF/RAG processing)",
        embedding_models_options,
        index=default_embed_index,
        key="embedding_model_select"
    )
    if "embed" not in st.session_state.selected_embedding_model.lower() and "nomic" not in st.session_state.selected_embedding_model.lower():
        st.sidebar.warning("It's recommended to use a dedicated embedding model (e.g., 'nomic-embed-text') for better results and consistent dimensions.")

else:
    st.sidebar.warning("No models found. Please ensure Ollama is running and models are downloaded (`ollama pull model_name`).")
    st.session_state.selected_llm_model = None
    st.session_state.selected_embedding_model = None

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="temperature_slider")

selected_option = st.sidebar.radio(
    "Choose your interaction mode:",
    ("Talk to LLM", "Talk based on RAG Memory", "Talk with PDF", "Create RAG Memory"),
    index=0 
)

if selected_option == "Talk to LLM":
    Direct_LLM_Chat(temperature)
elif selected_option == "Talk based on RAG Memory":
    RAG_Chat(temperature, FAISS_INDEX_PATH)
elif selected_option == "Talk with PDF":
    PDF_QA_Session(temperature)
elif selected_option == "Create RAG Memory":
    Create_RAG_Memory(FAISS_INDEX_PATH, RAG_DOCS_DIR)
