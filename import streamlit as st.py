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

# -------- Helpers --------
def pdf_to_text(file_path):
    try:
        pdf = PdfReader(file_path)
        return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
    except:
        return ""

def docx_to_text(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except:
        return ""

def file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

def mistral_generate(question, context, api_key, model="mistral-small-latest"):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are Geos Chatbot, a professional assistant for ground engineering reports."},
        {"role": "user", "content": f"Answer based only on this context:\n\n{context}\n\nQuestion: {question}"}
    ]
    data = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 512}
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError:
        if resp.status_code == 429:
            return "RATE_LIMIT"
        return f"❌ Mistral API error: {resp.text}"
    except Exception as e:
        return f"❌ Mistral API error: {repr(e)}"

def groq_generate(question, context, api_key, model="llama2-70b-4096"):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": "You are Geos Chatbot, a professional assistant for ground engineering reports."},
        {"role": "user", "content": f"Answer based only on this context:\n\n{context}\n\nQuestion: {question}"}
    ]
    data = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 512}
    try:
        resp = requests.post(url, headers=headers, json=data)
        resp.raise_for_status()
        result = resp.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Groq API error: {resp.text if 'resp' in locals() else repr(e)}"

# -------- Streamlit UI --------
st.set_page_config(page_title="Geos Chatbot", page_icon="🌍", layout="wide")
st.title("🌍 Geos Chatbot")
st.markdown("Search inside a single project or across **all projects**.")

mistral_key = st.secrets.get("MISTRAL_API_KEY", "")
groq_key = st.secrets.get("GROQ_API_KEY", "")

# -------- Project Discovery --------
root_path = st.text_input("Enter Root Folder Path (where all projects are stored):")
projects = []

if root_path and os.path.exists(root_path):
    projects = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    projects.sort()

if projects:
    project_options = ["All Projects"] + projects
    selected_project = st.selectbox("Select Project (or 'All Projects')", project_options)

    model_choice = st.selectbox(
        "Choose a Mistral model:",
        ["mistral-small-latest", "mistral-medium-latest", "mistral-large-latest"],
        index=0
    )

    query = st.text_input("🔎 Type your question:")
    if query:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        all_docs = []

        if selected_project != "All Projects":
            project_dir = os.path.join(INDEX_DIR, selected_project)
            if os.path.exists(project_dir):
                vector_store = FAISS.load_local(project_dir, embeddings, allow_dangerous_deserialization=True)
                retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                docs = retriever.get_relevant_documents(query)
                all_docs.extend(docs)
        else:
            for project in projects:
                project_dir = os.path.join(INDEX_DIR, project)
                if os.path.exists(project_dir):
                    vector_store = FAISS.load_local(project_dir, embeddings, allow_dangerous_deserialization=True)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
                    docs = retriever.get_relevant_documents(query)
                    all_docs.extend(docs)

        if all_docs:
            # Sort by relevance (score is implicit in retriever order, keep top 10 overall)
            all_docs = all_docs[:10]
            context = "\n".join([d.page_content for d in all_docs])

            with st.spinner("🤔 Thinking..."):
                answer = mistral_generate(query, context, mistral_key, model=model_choice)
                if answer == "RATE_LIMIT" and groq_key:
                    st.warning("⚠️ Mistral API rate limit reached. Switching to Groq...")
                    time.sleep(1)
                    answer = groq_generate(query, context, groq_key, model="llama2-70b-4096")

            st.subheader("📌 Answer")
            st.write(answer)

            st.subheader("📁 Relevant File Locations")
            file_paths = sorted(set([doc.metadata["source"] for doc in all_docs if "source" in doc.metadata]))
            for fp in file_paths:
                st.write(f"- {fp}")

            with st.expander("📑 Sources Preview"):
                for doc in all_docs:
                    st.write(f"📄 {doc.metadata['source']}")
                    st.write(doc.page_content[:300] + "...")
                    st.write("---")
        else:
            st.warning("⚠️ No relevant documents found.")
else:
    st.info("ℹ️ Enter a valid root path to discover projects.")
