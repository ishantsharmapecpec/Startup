import os
import json
import hashlib
import time
import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from PyPDF2 import PdfReader
import docx

INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)
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

# -------- LLM API Calls --------
def mistral_generate(question, context, api_key, model="mistral-small-latest"):
    """Call Mistral official API"""
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are Geos Chatbot, a professional assistant for ground engineering reports."},
        {"role": "user", "content": f"Answer based only on this project context:\n\n{context}\n\nQuestion: {question}"}
    ]
    data = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 512}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 429:
            return "RATE_LIMIT"
        return f"❌ Mistral API error: {response.text}"
    except Exception as e:
        return f"❌ Mistral API error: {repr(e)}"

def groq_generate(question, context, api_key, model="llama2-70b-4096"):
    """Fallback: Call Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are Geos Chatbot, a professional assistant for ground engineering reports."},
        {"role": "user", "content": f"Answer based only on this project context:\n\n{context}\n\nQuestion: {question}"}
    ]
    data = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 512}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Groq API error: {response.text if 'response' in locals() else repr(e)}"

# -------- Streamlit UI --------
st.set_page_config(page_title="Geos Chatbot", page_icon="🌍", layout="wide")

# 🎨 Styling with geotechnical wallpaper
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1501854140801-50d01698950b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80");
        background-size: cover;
        background-attachment: fixed;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
    }
    h1 {
        text-align: center;
        color: #00264d;
        font-weight: 900;
    }
    h2, h3, label {
        color: #003366 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Geos Chatbot")
st.markdown("Your intelligent assistant for **ground engineering reports**. Each project is kept separate to avoid mixing results.")

# -------- Admin Mode --------
admin_mode = False
admin_password = st.sidebar.text_input("Admin Password (optional)", type="password")

if "ADMIN_PASSWORD" in st.secrets and admin_password == st.secrets["ADMIN_PASSWORD"]:
    admin_mode = True
    st.sidebar.success("✅ Admin mode enabled")

# Load API keys from secrets
mistral_key = st.secrets.get("MISTRAL_API_KEY", "")
groq_key = st.secrets.get("GROQ_API_KEY", "")

if admin_mode:
    st.subheader("⚙️ Admin Controls")
    project_name = st.text_input("Enter Project Name (e.g., 5_Kingdom_Street)").strip().replace(" ", "_")

    uploaded_files = st.file_uploader(
        "📂 Upload one or more PDF/DOCX reports for this project",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    if project_name and uploaded_files and mistral_key and st.button("⚡ Process & Update Project Index"):
        project_dir = os.path.join(INDEX_DIR, project_name)
        os.makedirs(project_dir, exist_ok=True)
        processed_file_log = os.path.join(project_dir, "processed_files.json")

        if os.path.exists(processed_file_log):
            with open(processed_file_log, "r") as f:
                processed_files = json.load(f)
        else:
            processed_files = {}

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        if os.path.exists(os.path.join(project_dir, "index.faiss")):
            vector_store = FAISS.load_local(project_dir, embeddings, allow_dangerous_deserialization=True)
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

            vector_store.save_local(project_dir)
            with open(processed_file_log, "w") as f:
                json.dump(processed_files, f)
            st.success(f"✅ Index for project '{project_name}' updated successfully!")
        else:
            st.info("ℹ️ No new files to index.")

# ---- User Mode (Q&A by project) ----
projects = [f for f in os.listdir(INDEX_DIR) if os.path.isdir(os.path.join(INDEX_DIR, f))]
if projects and (mistral_key or groq_key):
    st.subheader("💬 Ask a Question")

    selected_project = st.selectbox("Select Project", projects)
    model_choice = st.selectbox(
        "Choose a Mistral model:",
        ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        index=0
    )

    query = st.text_input(f"🔎 Type your question about {selected_project}:")
    if query:
        project_dir = os.path.join(INDEX_DIR, selected_project)
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(project_dir, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])

        with st.spinner("🤔 Thinking..."):
            answer = mistral_generate(query, context, mistral_key, model=model_choice)

            if answer == "RATE_LIMIT" and groq_key:
                st.warning("⚠️ Mistral API rate limit reached. Switching to Groq...")
                time.sleep(1)
                answer = groq_generate(query, context, groq_key, model="llama2-70b-4096")

        st.subheader("📌 Answer")
        st.write(answer)

        with st.expander("📑 Sources"):
            for doc in docs:
                st.write(f"📎 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
