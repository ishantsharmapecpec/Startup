import os
import json
import hashlib
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
import docx

INDEX_DIR = "faiss_index"
METADATA_FILE = "processed_files.json"
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def mistral_generate(question, context, api_key, model="mistral-small-latest"):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are Geos Chatbot, a professional assistant for ground engineering reports."},
        {"role": "user", "content": f"Answer based on the context:\n\nContext:\n{context}\n\nQuestion: {question}"}
    ]
    data = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 512}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Mistral API error: {repr(e)}"

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_metadata(data):
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f)

# -------- Streamlit UI --------
st.set_page_config(page_title="Geos Chatbot", page_icon="🌍", layout="wide")

# 🎨 Custom CSS for wallpaper and styling
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1534081333815-ae5019106622?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 12px;
    }
    h1 {
        text-align: center;
        color: #003366;
        font-weight: 900;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Geos Chatbot")
st.markdown("Your intelligent assistant for **ground engineering reports**. Upload once, ask anytime!")

api_key = st.text_input("🔑 Enter your Mistral API Key", type="password")

model_choice = st.selectbox(
    "Choose a Mistral model:",
    ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
    index=0
)

uploaded_files = st.file_uploader(
    "📂 Upload one or more PDF/DOCX reports",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and api_key and st.button("⚡ Process & Update Index"):
    processed_files = load_metadata()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(INDEX_DIR):
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store = None

    new_chunks, new_metas = [], []
    updated = False

    for file in uploaded_files:
        file_bytes = file.read()
        file.seek(0)
        hash_val = file_hash(file_bytes)

        if hash_val in processed_files.values():
            st.warning(f"⚠️ Skipping already indexed file: {file.name}")
            continue

        text = pdf_to_text(file) if file.type == "application/pdf" else docx_to_text(file)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        new_chunks.extend(chunks)
        new_metas.extend([{"source": file.name}] * len(chunks))

        processed_files[file.name] = hash_val
        updated = True

    if updated:
        if vector_store:
            vector_store.add_texts(new_chunks, metadatas=new_metas)
        else:
            vector_store = FAISS.from_texts(new_chunks, embeddings, metadatas=new_metas)

        vector_store.save_local(INDEX_DIR)
        save_metadata(processed_files)
        st.success("✅ Index updated successfully!")
    else:
        st.info("ℹ️ No new files to index.")

# ---- Q&A ----
if os.path.exists(INDEX_DIR) and api_key:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    query = st.text_input("💬 Ask a question about your reports:")
    if query:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("🤔 Thinking..."):
            answer = mistral_generate(query, context, api_key, model=model_choice)

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📑 Sources"):
            for doc in docs:
                st.write(f"📎 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
