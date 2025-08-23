import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

st.title("📄 Ask Your PDF")
os.makedirs("data", exist_ok=True)

# --- Local models via Ollama ---
llm = Ollama(model="tinyllama")  # or "phi"
embed_model = OllamaEmbedding(model_name="nomic-embed-text")  # or "bge-m3"

# Set as globals so nothing falls back to OpenAI
Settings.llm = llm
Settings.embed_model = embed_model

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    file_path = "data/uploaded.pdf"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing your PDF..."):
        docs = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(docs, llm=llm)

    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Thinking..."):
            # Pass llm explicitly to avoid any default resolution
            query_engine = index.as_query_engine(llm=llm, response_mode="compact")
            response = query_engine.query(query)
        st.write(getattr(response, "response", response))
