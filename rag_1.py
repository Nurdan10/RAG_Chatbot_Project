import streamlit as st
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Set the permanent PDF file path
pdf_file_path = "N19-1423.pdf"

# Define the vectorstore directory
vectorstore_directory = "./vectorstore"

# Function to load or create ChromaDB connection and create index
@st.cache_resource
def load_or_create_chroma(file_path, vectorstore_dir):
    # Create embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=3072, openai_api_key=api_key)
    
    # Check if the vectorstore directory exists
    if os.path.exists(vectorstore_dir):
        # Load the existing Chroma index from the directory
        index = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        st.write("Loaded existing Chroma index from the directory.")
    else:
        # Load PDF document from the saved file path
        pdf_doc = read_doc(file_path)
        # Split the PDF content into chunks
        pdf_doc = chunk_data(docs=pdf_doc)
        # Create Chroma index from documents
        index = Chroma.from_documents(documents=pdf_doc, embedding=embeddings, persist_directory=vectorstore_dir)
        st.write("Created a new Chroma index and saved it to the directory.")
    
    return index

# PDF reading function for file path
def read_doc(file_path):
    # Use PyPDFium2Loader with the file path
    file_loader = PyPDFium2Loader(file_path)
    pdf_documents = file_loader.load()
    return pdf_documents

# Chunk splitting function
def chunk_data(docs, chunk_size=1000, chunk_overlap=200):
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    pdf = text_splitter.split_documents(docs)
    return pdf

# Function to retrieve relevant chunks for the given query
def retrieve_query(query, k=5, index=None):
    # Create a retriever to fetch relevant documents
    retriever = index.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# Summarization function for PDF
def summarize_pdf_content(pdf_path):
    pdf = read_doc(pdf_path)
    llm = ChatOpenAI(
        temperature=0,
        model_name='gpt-4o-mini',
        max_tokens=1024
    )
    chain = load_summarize_chain(
        llm,
        chain_type='stuff'
    )
    output_summary = chain.invoke(pdf)['output_text']
    return output_summary

# Function to generate answers based on the query
def get_answers(query, k=5, index=None):
    # Retrieve relevant documents
    doc_search = retrieve_query(query, k=k, index=index)
    # Define a template for the response
    template = """Use the following pieces of context to answer the user's question of "{question}".
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
"{context}" """
    # Set up the prompt template
    prompt_template = PromptTemplate(input_variables=['question', 'context'], template=template)
    # Initialize the language model
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, top_p=1)
    # Chain together the prompt and language model
    chain = prompt_template | llm | StrOutputParser()
    # Generate output using the chained prompt and model
    output = chain.invoke({"question": query, "context": doc_search})
    return output

# Function to inject custom CSS for sidebar
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* Change sidebar background color to a darker shade */
        .css-1l02x9p {
            background-color:#1e1e1e ; /* Very dark gray for the sidebar background */
        }
        /* Change sidebar text color to a lighter shade for better contrast */
        .css-1l02x9p .css-1v3fvcr {
            color: #dcdcdc; /* Light gray text color */
        }
        /* Adjust other elements if needed */
        .css-1l02x9p .css-1v3fvcr a {
            color: #87ceeb; /* Light blue color for links */
        }
        .css-1l02x9p .css-1v3fvcr a:hover {
            color: #b0e0e6; /* Lighter blue on hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Streamlit app layout
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Inject custom CSS
inject_custom_css()
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #ff000050;
    }
</style>
""", unsafe_allow_html=True)

# Navigation menu
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ("Chat", "Summarization"))

# Initialize or load previous Q&A from session state
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

# Load or create Chroma index
index = load_or_create_chroma(pdf_file_path, vectorstore_directory)

# Display based on selected page
if page == "Chat":
    st.title("Chat Functionality")

    def ask_question():
        with st.form(key='question_form', clear_on_submit=True):
            user_input = st.text_input("Please write your question here:")
            submit_button = st.form_submit_button("Ask")

            # When the user submits a question
            if submit_button and user_input:
                # First, try to find an answer from the document
                answer = get_answers(user_input, index=index)
                # If no answer found in the document, provide a default message
                if not answer or "don't know" in answer.lower():
                    answer_text = "The document doesn't contain an answer for this question."
                else:
                    answer_text = answer
                
                # Add new Q&A to the list
                st.session_state.qa_history.insert(0, {'question': user_input, 'answer': answer_text})
                
                # Limit the history to the last 3 items
                st.session_state.qa_history = st.session_state.qa_history[:]
                
                # Display the last 3 Q&A pairs
                for qa in st.session_state.qa_history:
                    st.write(f"**Question:** {qa['question']}")
                    st.write(f"**Answer:** {qa['answer']}")

    # Display the question box for user input
    ask_question()

elif page == "Summarization":
    st.title("Document Summarization")

    summary_button = st.button("Generate Summary")
    if summary_button:
        with st.spinner("Summarizing..."):
            summary = summarize_pdf_content(pdf_file_path)
            st.success("Summarization complete!")
            st.subheader("Summary")
            st.write(summary)