import os
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
import docx

INDEX_DIR = "faiss_index"
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"  # prevent Linux watch errors

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def together_generate(question, context, api_key, model="mistralai/Mistral-7B-Instruct-v0.1"):
    """Call Together.AI API for text generation (remote only)"""
    url = "https://api.together.xyz/inference"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.3,
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        if "output" in result and "choices" in result["output"]:
            return result["output"]["choices"][0]["text"].strip()
        else:
            return f"⚠️ Unexpected response: {result}"
    except Exception as e:
        return f"❌ Together API error: {repr(e)}"

# -------- Streamlit UI --------
st.title("📂 Project Q&A (FAISS + Together.AI)")

api_key = st.text_input("Enter your Together.AI API Key", type="password")

# Together-supported models (free tier)
model_choice = st.selectbox(
    "Choose a Together.AI model:",
    [
        "mistralai/Mistral-7B-Instruct-v0.1",
        "togethercomputer/llama-2-7b-chat",
        "togethercomputer/llama-2-13b-chat",
        "togethercomputer/llama-2-70b-chat",  # might need paid tier
    ],
    index=0
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files (only needed once by admin)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and api_key and st.button("Process & Save Index"):
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
if os.path.exists(INDEX_DIR) and api_key:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

    query = st.text_input("Ask a question about your documents:")
    if query:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("Thinking..."):
            answer = together_generate(query, context, api_key, model=model_choice)

        st.write("**Answer:**", answer)

        with st.expander("🔎 Sources"):
            for doc in docs:
                st.write(f"📌 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
