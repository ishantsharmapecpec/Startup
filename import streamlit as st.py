import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import docx, os

INDEX_DIR = "faiss_index"

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# -------- Streamlit UI --------
st.title("📂 Offline Project Q&A (FAISS + Ollama)")

uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files (only needed once)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Process & Save Index"):
    all_chunks, metadatas = [], []
    for file in uploaded_files:
        text = pdf_to_text(file) if file.type == "application/pdf" else docx_to_text(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        all_chunks.extend(chunks)
        metadatas.extend([{"source": file.name}] * len(chunks))

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(all_chunks, embeddings, metadatas=metadatas)

    vector_store.save_local(INDEX_DIR)
    st.success("✅ Knowledge base saved! You can now ask questions.")

# ---- Q&A ----
if os.path.exists(INDEX_DIR):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    query = st.text_input("Ask a question about your documents:")
    if query:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = Ollama(model="mistral")  # or "llama2"
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        with st.spinner("Thinking..."):
            result = qa(query)

        st.write("**Answer:**", result["result"])

        with st.expander("🔎 Sources"):
            for doc in result["source_documents"]:
                st.write(f"📌 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
