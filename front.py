import streamlit as st
import time
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

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

#session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_ready" not in st.session_state:
    st.session_state.api_ready = False


def check_api_health():
    #check if api is ready and running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("system_ready", False), data.get("ollama_running", False)
        return False, False
    except requests.exceptions.RequestException:
        return False, False

def query_api(question):
    #send quetion to api
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question":question, "stream":False},
            timeout=360
        )

        if response.status_code == 200:
            return response.json(), None
        else:
            error = response.json().get("detail", "Unknown error")
            return None, error
        
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Make sure it's running."
    except Exception as e:
        return None, str(e)

# Check API status
system_ready, ollama_running = check_api_health()
st.session_state.api_ready = system_ready

# Display status in sidebar
if system_ready:
    st.sidebar.success("API Ready")
elif ollama_running:
    st.sidebar.warning("Loading System...")
else:
    st.sidebar.error("API Not Running")
    st.sidebar.info("Start API with:\n`python api.py`")

st.sidebar.divider()

# Get system info
if system_ready:
    try:
        info_response = requests.get(f"{API_URL}/info", timeout=5)
        if info_response.status_code == 200:
            info = info_response.json()
            st.sidebar.metric("Documents", info.get("document_count", 0))
    except:
        pass

st.sidebar.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"{meta.get('category', 'N/A')}")
            with col2:
                st.caption(f"{meta.get('confidence', 0):.1%}")
            with col3:
                st.caption(f"{meta.get('latency', 0):.2f}s")

# Chat input
if system_ready:
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Generate answer
        with st.spinner("Generating answer..."):
            result, error = query_api(user_query)
            
            if error:
                st.error(f"Error: {error}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {error}"
                })
            else:
                answer = result['answer']
                
                # Display assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": {
                        "category": result['category'],
                        "confidence": result['confidence'],
                        "latency": result['latency']
                    }
                })
                
                # Rerun to show the message with metadata
                st.rerun()
else:
    st.warning("API is not ready. Please start the API server:")
    st.code("python api.py", language="bash")
    
    if st.button("Retry Connection"):
        st.rerun()
