from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering.chain import load_qa_chain

import streamlit as st
import fitz

from dotenv import load_dotenv
load_dotenv()
import os

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "pdf"
index = pc.Index(index_name)

# Initialize Embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="gemini-2.0-flash-001")

## Lets Read the Localdocument
def read_local(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

## Divide the localdocs into chunks
def chunk_local(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc

# Function: Chunk text
def chunk_text(text, chunk_size=800, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    docs = splitter.create_documents([text])
    return docs

# Function: Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# Function: Create vector store
def create_vectorstore(documents):
    vectorstore = PineconeVectorStore.from_documents(
        documents,
        embedding=embeddings,
        index_name=index_name
    )
    return vectorstore

# Function: Retrieve relevant docs
def retrieve_docs(vectorstore, query):
    return vectorstore.similarity_search(query=query)

# Function: Answer question
def get_answer(docs, query):
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    answer = qa_chain.run(input_documents=docs, question=query)
    return answer

# Function: Answer question
def get_local_answer():
    documents=chunk_local(docs=read_local('documents/'))
    query= 'Whats the ph of milk?'
    vs = create_vectorstore(documents)
    retrieved_docs = retrieve_docs(vs, query)
    answer = get_answer(retrieved_docs, query)
    print(answer)

#get_local_answer('query')

# Streamlit UI
st.set_page_config(page_title="PDF Q&A RAG", layout="wide")
st.title("Chat with PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Extract text
    raw_text = extract_text_from_pdf(uploaded_file)

    # Chunk text
    docs = chunk_text(raw_text)
    st.write(f"✅ Split into {len(docs)} chunks")

    # Create vector store
    vectorstore = create_vectorstore(docs)
    st.success("✅ Document indexed in Pinecone")

    # Input for question
    query = st.text_input("Ask a question about this PDF:")

    if query:
        with st.spinner("Searching and generating answer..."):
            # Retrieve similar chunks
            retrieved_docs = retrieve_docs(vectorstore, query)

            # Generate answer
            answer = get_answer(retrieved_docs, query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Top Matching Chunks")
        for i, doc in enumerate(retrieved_docs, 1):
            st.markdown(f"**Chunk {i}:** {doc.page_content[:300]}...")

else:
    st.info("Upload a PDF to get started.")

