import os
import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def Process_Documents_To_Vector_Store(documents, embedding_model_name, ollama_url, progress_text="Processing documents..."):
    """
    Processes a list of documents into chunks, generates embeddings, and builds a FAISS vector store.
    """
    if not embedding_model_name:
        st.error("No embedding model selected. Cannot process documents.")
        return None

    try:
        with st.spinner(progress_text):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            st.info(f"Split documents into {len(texts)} chunks.")

            embeddings = OllamaEmbeddings(base_url=ollama_url, model=embedding_model_name)
            vector_store = FAISS.from_documents(texts, embeddings)
            st.success("Embeddings generated and vector database built!")
            return vector_store
    except Exception as e:
        st.error(f"Error during document processing or embedding generation: {e}. "
                 "Please ensure the selected Ollama embedding model is available and downloaded (`ollama pull {embedding_model_name}`).")
        return None



def Create_RAG_Memory(FAISS_INDEX_PATH, RAG_DOCS_DIR):
    st.header("Create RAG Memory (Persistent)")
    if not st.session_state.selected_embedding_model:
        st.warning("Please select an Embedding Model in the sidebar.")
        return

    st.markdown("""
    Upload text files (.txt) or PDF documents here to build a persistent RAG memory.
    This memory will be saved locally as `faiss_index.bin` and can be used in the 'Talk based on RAG Memory' option.
    """)

    uploaded_rag_files = st.file_uploader(
        "Upload documents (PDF, TXT) for RAG memory",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="rag_memory_uploader"
    )

    if st.button("Build/Update RAG Memory", type="primary", key="build_rag_button"):
        if not uploaded_rag_files:
            st.warning("Please upload some documents to build the RAG memory.")
            return

        all_documents = []
        temp_files = [] 

        for uploaded_file in uploaded_rag_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=RAG_DOCS_DIR) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                temp_files.append(tmp_file_path) 

            try:
                if file_extension == ".pdf":
                    loader = PyPDFLoader(tmp_file_path)
                elif file_extension == ".txt":
                    loader = TextLoader(tmp_file_path)
                else:
                    st.warning(f"Skipping unsupported file type: {uploaded_file.name}")
                    continue
                
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    doc.metadata['source'] = uploaded_file.name
                all_documents.extend(loaded_docs)
                st.info(f"Loaded {len(loaded_docs)} pages/sections from {uploaded_file.name}")

            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                continue
        
        if not all_documents:
            st.warning("No valid documents were loaded to build RAG memory.")
            return

        vector_store = Process_Documents_To_Vector_Store(
            all_documents,
            st.session_state.selected_embedding_model,
            st.session_state.ollama_base_url,
            "Generating embeddings and building persistent RAG memory..."
        )

        if vector_store:
            st.session_state.rag_vector_store = vector_store
            st.session_state.rag_memory_loaded = True
            try:
                vector_store.save_local(FAISS_INDEX_PATH)
                st.success(f"RAG memory successfully built and saved to '{FAISS_INDEX_PATH}'!")
            except Exception as e:
                st.error(f"Error saving FAISS index to disk: {e}")
        else:
            st.session_state.rag_vector_store = None
            st.session_state.rag_memory_loaded = False
        
        for f_path in temp_files:
            if os.path.exists(f_path):
                os.remove(f_path)
    
    st.markdown("---")
    st.subheader("Current RAG Memory Status")
    if st.session_state.rag_memory_loaded:
        st.success("Persistent RAG memory is loaded and ready!")
        st.write(f"Vector store type: {type(st.session_state.rag_vector_store)}")
    else:
        st.info("No persistent RAG memory loaded. Please build it above.")
    
    if os.path.exists(FAISS_INDEX_PATH):
        st.info(f"FAISS index file found on disk: `{FAISS_INDEX_PATH}`. It will be loaded automatically when you select 'Talk based on RAG Memory'.")
        if st.button("Delete Persistent RAG Memory File", key="delete_rag_file_button"):
            try:
                os.remove(FAISS_INDEX_PATH)
                st.session_state.rag_vector_store = None
                st.session_state.rag_memory_loaded = False
                st.success("Persistent RAG memory file deleted.")
                st.rerun()
            except Exception as e:
                st.error(f"Error deleting FAISS index file: {e}")



