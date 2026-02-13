import streamlit as st
import time
from rag import QASystem, check_ollama

st.set_page_config(
    page_title="Document Q&A",
    layout="centered"
)

st.title("Document Q&A system")
st.caption("Ask question about your document using RAG")

st.sidebar.header("Settings")

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages =[]
    st.rerun()

st.sidebar.divider()

if "qa_system" not in st.session_state:
    st.session_state.qa_system = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

#Intitialize system
if not st.session_state.system_ready:
    with st.spinner("Initializing System..."):
        try:
            if not check_ollama():
                st.error("Ollama is not running. Start with 'ollama serve'")
                st.stop()
            st.session_state.qa_system = QASystem()
            st.session_state.system_ready = True
        except Exception as e:
            st.error(f"Failed to initialize system: {str(e)}")
            st.stop()

#show system status in sidebar
if st.session_state.system_ready:
    st.sidebar.success("System Ready!")
else:
    st.sidebar.warning("Loading...")

st.sidebar.divider()

#display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

#chat input
if st.session_state.system_ready:
    user_query = st.chat_input("Ask your question")

    if user_query:
        st.session_state.messages.append({"role":"user", "content":user_query})
        st.chat_message("user").write(user_query)

        with st.spinner("Generating answer..."):
            try:
                result = st.session_state.qa_system.query(user_query)
                answer = result['answer']

                st.session_state.messages.append({"role":"assistant", "content": answer})
                st.chat_message("assistant").write(answer)
            except Exception as e:
                error_msg = f"Error: {str(e)}" 
                st.error(error_msg)
                st.session_state.messages.append({"role":"assistant", "content":error_msg})
else:
    st.info("Intializing system, please wait...")