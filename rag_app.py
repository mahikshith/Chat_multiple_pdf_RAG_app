import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import transformers

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
import getpass

inference_api_key = getpass.getpass("HF_TOKEN")
# there is a problem with HF_TOKEN

load_dotenv()
os.getenv("GOOGLE_API_KEY")
HF_API_TOKEN = os.getenv("HF_TOKEN")
transformers.set_access_token(HF_API_TOKEN)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.create_documents(text)
    return chunks




def create_vectorstore(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs) 
    vectorstore = FAISS.from_documents(text_chunks, embeddings)
    vectorstore.save_local("mykee_faiss_index")
    return vectorstore

def create_qa_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001",temperature=0.5)
    prompt = ChatPromptTemplate(
        input_variables=["user_input", "context"],
        template="""Use the context given to answer the question in a detailed format as possible
        make sure to provide all the details and include bullet points while answering, if the answer is not in
        provided context just say, "Answer is not available in the context", don't provide the wrong answer
        \n\nContext:\n{context}\n\nQuestion:\n{user_input}""")
    
    retrieval_chain = RetrievalQA.from_llm_vectorstores(llm, vectorstore, prompt=prompt)
    return retrieval_chain

def main():
    st.title("RAG App - Chat with PDFs")

    uploaded_files = st.sidebar.file_uploader("Upload PDFs and click submit", type=["pdf"],accept_multiple_files=True)

    if uploaded_files:
        text = get_pdf_text(uploaded_files)
        
        docs = split_text(text)
        vectorstore = create_vectorstore(docs)
        qa_chain = create_qa_chain(vectorstore)

        query = st.text_input("Ask a question about any information from the PDFs")
        if query:
            response = qa_chain.run(query)
            st.text_area("Answer:", value=response)

if __name__ == "__main__":
    main()