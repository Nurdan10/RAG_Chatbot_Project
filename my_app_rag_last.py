import streamlit as st
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from PIL import Image
import hashlib

# Load environment variables
load_dotenv()

# Get API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI model
llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

# Create LLMChain for Question-Answering
qa_prompt = PromptTemplate(input_variables=["question", "docs_content"], template="Based on the following content:\n{docs_content}\n\nQ: {question}\nA:")
qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# Create LLMChain for Summarization
summarize_prompt = PromptTemplate(input_variables=["docs_content"], template="Summarize the following content:\n{docs_content}\n\nSummary:")
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# Streamlit application
st.markdown("<h1 style='text-align: center;'>Welcome to the RAG Chatbot and Summarization Project App!</h1>", unsafe_allow_html=True)
img = Image.open("images/pdf_img.jpg")
st.image(img, use_column_width=True)
st.write("Upload a PDF, ask questions, and get summaries!")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# Vector store directory
vectorstore_dir = "vectorstore"
docs = None  # Initialize docs variable

if uploaded_file is not None:
    # Read and hash the PDF content
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Create a hash of the PDF content
    pdf_hash = hashlib.md5(text.encode()).hexdigest()
    
    # Path to store the hash
    hash_file_path = os.path.join(vectorstore_dir, "pdf_hash.txt")
    
    # Check if the vector store and hash exist
    if os.path.exists(vectorstore_dir) and os.path.exists(hash_file_path):
        with open(hash_file_path, "r") as hash_file:
            stored_hash = hash_file.read().strip()
        
        # If the hash matches, load the existing vectorstore
        if stored_hash == pdf_hash:
            st.write("Loading existing vector store...")
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        else:
            # If the hash doesn't match, create a new vectorstore
            st.write("PDF has changed. Rebuilding vector store...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_text(text)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectorstore = Chroma.from_texts(docs, embeddings, persist_directory=vectorstore_dir)
            vectorstore.persist()
            # Save the new hash
            with open(hash_file_path, "w") as hash_file:
                hash_file.write(pdf_hash)
    else:
        # If no vectorstore exists, create one and save the hash
        st.write("No existing vector store found. Creating a new one...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(text)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_texts(docs, embeddings, persist_directory=vectorstore_dir)
        vectorstore.persist()
        # Save the hash
        os.makedirs(vectorstore_dir, exist_ok=True)
        with open(hash_file_path, "w") as hash_file:
            hash_file.write(pdf_hash)

    # Ensure docs is defined for summarization
    if docs is None:
        # If docs was not defined due to loading an existing vector store, split the text now
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(text)

    # Choose mode: Question-Answering or Summarization
    mode = st.selectbox("Choose a mode:", ["Question-Answering", "Summarization"])

    if mode == "Question-Answering":
        # Question-Answering mode
        question = st.text_input("Ask a question based on the PDF:")
        if question:
            relevant_docs = vectorstore.similarity_search(question)
            docs_content = "\n".join([doc.page_content for doc in relevant_docs])
            answer = qa_chain.run(question=question, docs_content=docs_content)
            st.write("Answer:", answer)

    elif mode == "Summarization":
        # Summarization mode
        batch_size = 5  # Process 5 chunks at a time
        intermediate_summaries = []

        for i in range(0, len(docs), batch_size):
            batch_docs_content = "\n".join(docs[i:i+batch_size])
            batch_summary = summarize_chain.run(docs_content=batch_docs_content)
            intermediate_summaries.append(batch_summary)

        # Combine the intermediate summaries and summarize again
        final_docs_content = "\n".join(intermediate_summaries)
        final_summary = summarize_chain.run(docs_content=final_docs_content)
        st.write("Summary:", final_summary)

