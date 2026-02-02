import os
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import ollama

def Load_Persistent_RAG_Memory(FAISS_INDEX_PATH):
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            embeddings_instance = OllamaEmbeddings(
                base_url=st.session_state.ollama_base_url,
                model=st.session_state.selected_embedding_model
            )
            st.session_state.rag_vector_store = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings_instance, allow_dangerous_deserialization=True
            )
            st.session_state.rag_memory_loaded = True
            st.success("RAG memory loaded successfully from disk!")
        except Exception as e:
            st.error(f"Error loading RAG memory from disk: {e}. It might be corrupted or created with a different embedding model. Please try recreating it.")
            st.session_state.rag_vector_store = None
            st.session_state.rag_memory_loaded = False
    else:
        st.session_state.rag_vector_store = None


def RAG_Chat(temperature, FAISS_INDEX_PATH):
    st.header("Talk based on RAG Memory")
    
    if not st.session_state.selected_llm_model or not st.session_state.selected_embedding_model:
        st.warning("Please select both LLM and Embedding Models in the sidebar.")
        return

    if not st.session_state.rag_memory_loaded:
        with st.spinner("Attempting to load RAG memory..."):
            Load_Persistent_RAG_Memory(FAISS_INDEX_PATH)
    
    if st.session_state.rag_vector_store is None:
        st.info("No RAG memory found. Please go to 'Create RAG Memory' to build it.")
        return

    if "rag_chat_messages" not in st.session_state:
        st.session_state.rag_chat_messages = []

    for message in st.session_state.rag_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about your RAG memory...", key="rag_chat_input"):
        st.session_state.rag_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Searching RAG memory and thinking..."):
            try:
                retriever = st.session_state.rag_vector_store.as_retriever()
                docs = retriever.get_relevant_documents(prompt)

                context_text = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = f"Answer the question based on the following context:\n\n{context_text}\n\nQuestion: {prompt}\nAnswer:"

                full_response = ""
                placeholder = st.empty()
                for chunk in ollama.chat(
                    model=st.session_state.selected_llm_model,
                    messages=[{"role": "user", "content": final_prompt}],
                    stream=True,
                ):
                    if "message" in chunk and "content" in chunk["message"]:
                        token = chunk["message"]["content"]
                        full_response += token
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)
                st.session_state.rag_chat_messages.append({"role": "assistant", "content": full_response})
                
                if docs:
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(docs):
                            st.write(f"**Document {i+1}:**")
                            st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
                            st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                            st.write(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"Error getting answer from RAG: {e}")
                st.session_state.rag_chat_messages.append({
                    "role": "assistant",
                    "content": "Sorry, I encountered an error while retrieving from RAG memory."
                })