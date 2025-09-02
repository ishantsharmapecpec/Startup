import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
import docx

INDEX_DIR = "faiss_index"

# ---- Prevent inotify watch errors on Linux/Streamlit Cloud ----
os.environ["STREAMLIT_SERVER_FILEWATCHERTYPE"] = "none"

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def hf_generate(question, context, hf_key, model="google/flan-t5-base"):
    """Query Hugging Face Inference API only (no local model loading)."""
    client = InferenceClient(model=model, token=hf_key)
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = client.text_generation(
        prompt,
        max_new_tokens=256,
        temperature=0.3,
        do_sample=False,
        return_full_text=False,
    )
    return response

# -------- Streamlit UI --------
st.title("📂 Project Q&A (FAISS + Hugging Face Inference API)")

hf_key = st.text_input("Enter your Hugging Face API Key", type="password")

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
            answer = hf_generate(query, context, hf_key, model="google/flan-t5-base")

        st.write("**Answer:**", answer)

        with st.expander("🔎 Sources"):
            for doc in docs:
                st.write(f"📌 {doc.metadata['source']}")
                st.write(doc.page_content[:300] + "...")
                st.write("---")
