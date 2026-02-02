import tempfile
import os
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_memory import Process_Documents_To_Vector_Store

import streamlit as st
import tempfile
import os
import ollama
from langchain_community.document_loaders import PyPDFLoader

def PDF_QA_Session(temperature):
    st.header("Talk with PDF (Temporary Memory)")

    if not st.session_state.selected_llm_model or not st.session_state.selected_embedding_model:
        st.warning("Please select both LLM and Embedding Models in the sidebar.")
        return

    if st.session_state.pdf_processed_for_qa:
        st.info(f"PDF '{st.session_state.uploaded_pdf_name}' already processed. Ask questions below.")
        if st.button("Upload New PDF / Clear Current PDF Memory", key="clear_pdf_qa_button"):
            st.session_state.pdf_processed_for_qa = False
            st.session_state.pdf_vector_store = None
            st.session_state.uploaded_pdf_name = None
            st.rerun()

    if not st.session_state.pdf_processed_for_qa:
        uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"], key="pdf_qa_uploader")
        if uploaded_file is not None:
            st.session_state.uploaded_pdf_name = uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            st.success(f"PDF uploaded successfully: {uploaded_file.name}")

            try:
                loader = PyPDFLoader(tmp_file_path)
                documents = loader.load()

                vector_store = Process_Documents_To_Vector_Store(
                    documents,
                    st.session_state.selected_embedding_model,
                    st.session_state.ollama_base_url,
                    "Loading and processing PDF for Q&A..."
                )

                if vector_store:
                    st.session_state.pdf_vector_store = vector_store
                    st.session_state.pdf_processed_for_qa = True
                    st.rerun()
                else:
                    st.session_state.pdf_vector_store = None
                    st.session_state.pdf_processed_for_qa = False
            except Exception as e:
                st.error(f"An error occurred during PDF processing: {e}")
                st.session_state.pdf_vector_store = None
                st.session_state.pdf_processed_for_qa = False
            finally:
                if os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)

    if st.session_state.pdf_processed_for_qa and st.session_state.pdf_vector_store is not None:
        question = st.text_area("Enter your question about the PDF:", height=100, key="pdf_qa_question_input")

        if st.button("Get Answer from PDF", key="get_pdf_answer_button"):
            if question:
                with st.spinner("Fetching answer from PDF..."):
                    try:
                        retriever = st.session_state.pdf_vector_store.as_retriever()
                        docs = retriever.get_relevant_documents(question)

                        context_text = "\n\n".join([doc.page_content for doc in docs])
                        final_prompt = f"Answer the question based on the following PDF context:\n\n{context_text}\n\nQuestion: {question}\nAnswer:"

                        placeholder = st.empty()
                        full_response = ""
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

                        if docs:
                            st.subheader("Source Documents:")
                            for i, doc in enumerate(docs):
                                st.write(f"**Document {i+1}:**")
                                st.write(f"Source: {doc.metadata.get('source', 'N/A')}")
                                st.write(f"Page: {doc.metadata.get('page', 'N/A')}")
                                st.write(doc.page_content[:500] + "...")

                    except Exception as e:
                        st.error(f"Error getting answer from PDF: {e}. Please ensure the selected Ollama models are running and configured correctly.")
            else:
                st.warning("Please enter a question.")
    elif not st.session_state.pdf_processed_for_qa:
        st.info("Upload a PDF document above to enable Q&A.")
