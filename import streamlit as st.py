import os
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
import docx

INDEX_DIR = "faiss_index"

# ---- Prevent file watcher crash on Linux/Streamlit Cloud ----
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def hf_generate(question, context, hf_key, model="bigscience/bloomz-560m"):
    """Query Hugging Face Inference API via HTTP"""
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {hf_key}"}
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()
        result = response.json()
        # Hugging Face API sometimes returns a dict, sometimes a list
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"❌ Hugging Face API error: {result['error']}"
        else:
            return f"⚠️ Unexpected response format: {result}"
    except Exception as e:
        return f"❌ Hugging Face API error: {repr(e)}"

# -------- Streamlit UI --------
st.title("📂 Project Q&A (FAISS + Hugging Face Inference API)")

hf_key = st.text_input("Enter your Hugging Face API Key", type="password")

# ✅ Limit to free models with Hosted Inference API
model_choice = st.selectbox(
    "Choose a Hugging Face model:",
    [
        "bigscience/bloomz-560m", # ✅ safe default
        "google/flan-t5-base",    # may work (sometimes Pro-only)
        "google/flan-t5-large",   # likely Pro-only
    ],
    index=0
)

uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files (only needed once by admin)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and hf_key and st.button("Process & Save Index"):
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
if os.path.exists(INDEX_DIR) and hf_key:
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
            answer = hf_generate(query, context, hf_key, model=model_choice)

        st.write("**Answer:**", answer)

        with st.expander("🔎 Sources"):
            for doc in docs:
                st.write(f"📌 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
