import os
import base64
import streamlit as st

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- basic page setup ---
st.set_page_config(page_title="Ask Your PDF", page_icon="📄", layout="wide")
os.makedirs("data", exist_ok=True)

# --- background image (put your file at assets/background1.png) ---
def set_background(image_path="assets/background1.png"):
    if not os.path.exists(image_path):
        return
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:{mime};base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        .card {{
            background: rgba(0,0,0,0.55);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background()

# --- sidebar: simple controls ---
st.sidebar.header("Settings")
model = st.sidebar.selectbox("LLM (Ollama)", ["tinyllama", "phi"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
embed_name = st.sidebar.selectbox("Embeddings", ["nomic-embed-text", "bge-m3"], index=0)
if st.sidebar.button("🧹 Clear chat"):
    st.session_state.pop("chat", None)

# --- init models (local, no API keys) ---
Settings.llm = Ollama(model=model, temperature=temperature, request_timeout=120)
Settings.embed_model = OllamaEmbedding(model_name=embed_name)

# --- title & blurb ---
st.markdown('<div class="card"><h1>📄 Ask Your PDF</h1><p>Upload a PDF and ask questions. Runs locally with Ollama.</p></div>', unsafe_allow_html=True)

# --- upload & (re)index when file changes ---
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded:
    if uploaded.name != st.session_state.get("last_file"):
        with open("data/uploaded.pdf", "wb") as f:
            f.write(uploaded.read())
        st.session_state.last_file = uploaded.name

        with st.spinner("Indexing…"):
            docs = SimpleDirectoryReader("data").load_data()
            st.session_state.index = VectorStoreIndex.from_documents(docs, llm=Settings.llm)
            st.session_state.query_engine = st.session_state.index.as_query_engine(
                llm=Settings.llm, response_mode="compact"
            )
        st.success(f"Indexed: {uploaded.name}")

# --- simple chat UI ---
if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(content)

q = st.chat_input("Ask a question about your PDF…")
if q:
    st.session_state.chat.append(("user", q))
    with st.chat_message("user"):
        st.markdown(q)

    if "query_engine" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("Please upload a PDF first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = st.session_state.query_engine.query(q)
            answer = getattr(resp, "response", str(resp))
            st.markdown(answer)
            st.session_state.chat.append(("assistant", answer))
