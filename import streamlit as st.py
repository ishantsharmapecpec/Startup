import streamlit as st
import pdfplumber
from docx import Document
from langchain.text_splitter import CharacterTextSplitter

# Updated imports from langchain_community
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Helper functions ---
def pdf_to_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def docx_to_text(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# --- Streamlit App ---
st.title("GeoReport Q&A AI")
st.write("Upload your geotechnical report (PDF or Word) and ask questions!")

uploaded_file = st.file_uploader("Upload a report", type=["pdf", "docx"])
api_key = st.text_input("Enter your OpenAI API Key:", type="password")

if uploaded_file and api_key:
    # Convert file to text
    if uploaded_file.type == "application/pdf":
        text = pdf_to_text(uploaded_file)
    else:
        text = docx_to_text(uploaded_file)

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Create embeddings and Chroma vector store
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = Chroma.from_texts(chunks, embedding=embeddings)

    # Create retrieval-based Q&A
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini"),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    # Ask questions
    question = st.text_input("Ask a question about the report:")
    if question:
        with st.spinner("Generating answer..."):
            answer = qa.run(question)
        st.write("**Answer:**", answer)
