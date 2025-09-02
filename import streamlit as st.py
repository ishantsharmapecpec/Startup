# pip install -r requirements.txt
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import docx

# -------- Helpers --------
def pdf_to_text(file):
    pdf = PdfReader(file)
    text = ""
    for i, page in enumerate(pdf.pages, start=1):
        try:
            text += f"\n--- Page {i} ---\n" + page.extract_text()
        except:
            pass
    return text

def docx_to_text(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

# -------- Streamlit UI --------
st.title("📂 Multi-Document Q&A (Streamlit + FAISS)")

api_key = st.text_input("Enter your OpenAI API Key", type="password")
uploaded_files = st.file_uploader(
    "Upload one or more PDF/DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files and api_key and st.button("Process Files"):
    all_chunks, metadatas = [], []

    for file in uploaded_files:
        if file.type == "application/pdf":
            text = pdf_to_text(file)
        else:
            text = docx_to_text(file)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)

        all_chunks.extend(chunks)
        metadatas.extend([{"source": file.name}] * len(chunks))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    # Build FAISS index (no sqlite issues!)
    vector_store = FAISS.from_texts(all_chunks, embedding=embeddings, metadatas=metadatas)

    st.session_state["vector_store"] = vector_store
    st.success("✅ Documents processed! You can now ask questions.")

if "vector_store" in st.session_state:
    query = st.text_input("Ask a question about your documents:")
    if query:
        retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 3})

        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini"),
            retriever=retriever,
            return_source_documents=True
        )

        with st.spinner("Finding the answer..."):
            result = qa(query)
            st.write("**Answer:**", result["result"])

            with st.expander("🔎 Sources"):
                for doc in result["source_documents"]:
                    st.write(f"📌 {doc.metadata['source']}")
                    st.write(doc.page_content[:300] + "...")
                    st.write("---")
