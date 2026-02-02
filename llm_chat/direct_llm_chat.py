import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

def Direct_LLM_Chat(temperature):
    st.header("Talk to LLM Directly")
    if not st.session_state.selected_llm_model:
        st.warning("Please select an LLM (Question Answering) Model in the sidebar.")
        return

    llm = Ollama(
        base_url=st.session_state.ollama_base_url,
        model=st.session_state.selected_llm_model,
        temperature=temperature,
    )

    if "direct_chat_messages" not in st.session_state:
        st.session_state.direct_chat_messages = []
    for message in st.session_state.direct_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Say something...", key="direct_chat_input"):
        st.session_state.direct_chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    placeholder = st.empty()
                    streamed_response = ""
                    for chunk in llm.stream(prompt):
                        streamed_response += chunk
                        placeholder.markdown(streamed_response + "▌")  # Animated effect

                    placeholder.markdown(streamed_response)  # Finalize
                    st.session_state.direct_chat_messages.append({"role": "assistant", "content": streamed_response})
                except Exception as e:
                    st.error(f"Error communicating with LLM: {e}")
                    st.session_state.direct_chat_messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
