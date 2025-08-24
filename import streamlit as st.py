# Install required packages (make sure to include in requirements.txt)
# pip install streamlit langchain langchain-community openai PyPDF2 python-docx chromadb tiktoken

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
import docx

# Function to extract text from PDF
def pdf_to_text(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# Function to extract text from DOCX
def docx_to_text(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

st.title("📄 Report Question Answering App")

# Upload file and provide OpenAI API key
uploaded_file = st.file_uploader("Upload your PDF or DOCX report", type=["pdf", "docx"])
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_file and api_key:
    if 'qa' not in st.session_state:
        # Convert file to text
        if uploaded_file.type == "application/pdf":
            text = pdf_to_text(uploaded_file)
        else:
            text = docx_to_text(uploaded_file)

        # Split text and create vector store
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vector_store = Chroma.from_texts(chunks, embedding=embeddings)

        # Create RetrievalQA chain and store in session state
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=api_key, model_name="gpt-4o-mini"),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        st.session_state['qa'] = qa
        st.success("✅ File processed! You can now ask questions.")

# Ask questions only if QA object exists
if 'qa' in st.session_state:
    question = st.text_input("Ask a question about the report:")
    if question:
        with st.spinner("Generating answer..."):
            answer = st.session_state['qa'].run(question)
        st.write("**Answer:**", answer)
