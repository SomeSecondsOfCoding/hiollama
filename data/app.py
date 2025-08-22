import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import Ollama

st.title("📄 Ask Your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    #save file 
    with open("data/uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())


        # Load and Index Pdf

        docs = SimpleDirectoryReader("data").load_data()
        llm = Ollama(model= "tinyllama") 
        index = VectorStoreIndex.from_documents(docs, llm=llm)
        query_engine = index.as_query_engine()


        #Query
        query = st.text_input("Ask a question about your PDF:")
        if query:
            response = query_engine.query(query)
            st.write(response)