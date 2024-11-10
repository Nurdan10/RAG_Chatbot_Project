import streamlit as st
from PyPDF2 import PdfReader
from langchain import OpenAI
from langchain.chains import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0, openai_api_key="YOUR_OPENAI_API_KEY")

# Set up the Streamlit app layout
st.title("RAG Chatbot with PDF Summarization")
st.write("Upload a PDF, ask questions, and get summaries!")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split the text into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(text)

    # Initialize embeddings and Chroma vectorstore
    embeddings = OpenAIEmbeddings(openai_api_key="YOUR_OPENAI_API_KEY")
    vectorstore = Chroma.from_texts(docs, embeddings)

    # Choose a mode: question-answering or summarization
    mode = st.selectbox("Choose a mode:", ["Question-Answering", "Summarization"])

    if mode == "Question-Answering":
        # Question-answering mode
        question = st.text_input("Ask a question based on the PDF:")
        if question:
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            relevant_docs = vectorstore.similarity_search(question)
            answer = qa_chain.run(input_documents=relevant_docs, question=question)
            st.write("Answer:", answer)

    elif mode == "Summarization":
        # Summarization mode
        summarize_chain = load_qa_chain(llm, chain_type="map_reduce")
        summary = summarize_chain.run(input_documents=docs, question="Summarize the PDF content")
        st.write("Summary:", summary)

