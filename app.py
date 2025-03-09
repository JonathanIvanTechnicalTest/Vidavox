import os
import psycopg2
import pdfplumber
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

MODEL_NAME = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

DB_CONFIG = {
    "dbname": "Robot",
    "user": "postgres",       # Change this
    "password": "Kaikuzu1@",  # Change this
    "host": "localhost"
}

def fetch_postgres_data():
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM newrobotlogs limit 5")  # Modify as needed
        data = cursor.fetchall()
        cursor.close()
        connection.close()
        return data
    except Exception as e:
        st.error(f"❌ PostgreSQL Connection Failed: {e}")
        return []

def process_pdfs(pdf_folder):
    """Extract text from PDFs in the given folder"""
    text_data = ""
    if not os.path.exists(pdf_folder):
        st.error(f"❌ Folder '{pdf_folder}' not found!")
        return ""

    files = os.listdir(pdf_folder)
    if not files:
        st.error(f"❌ No PDFs found in '{pdf_folder}'")
        return ""

    for file in files:
        if file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(pdf_folder, file)) as pdf:
                for page in pdf.pages:
                    text_data += page.extract_text() or ""  # Handle None values
    return text_data.strip()

pdf_text = process_pdfs("pdfs")
postgres_data = fetch_postgres_data()
documents = pdf_text + "\n" + "\n".join([str(row) for row in postgres_data])

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(documents)

vectorstore = FAISS.from_texts(chunks, embeddings)

def clean_retrieved_text(text):
    unwanted_phrases = [
        "Dokumentasi Produk Robot X-2000", "Dokumentasi Produk"
    ]
    
    for phrase in unwanted_phrases:
        text = text.replace(phrase, "")
        
    text = re.sub(r'\bBab \d+(:?.*)?', '', text)
    
    text = re.sub(r'Halaman\s*\d{1,3}', '', text) 

    text = re.sub(r'\s{2,}', ' ', text)
    
    return text.strip()

def retrieve_info(query):
    results = vectorstore.similarity_search(query, k=3)
    cleaned_results = [clean_retrieved_text(res.page_content) for res in results]
    return cleaned_results

# ✅ Streamlit App
def main():
    st.title("🤖 RAG Chatbot: Robot X2000")
    st.write("Ask me anything about the Robot X2000!")

    # Input for user query
    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Retrieving relevant information..."):
            retrieved_docs = retrieve_info(user_query)
            context = "\n\n".join(retrieved_docs)  
            
        st.subheader("📌 Answer:")

        formatted_context = context.replace("\n", "  \n")  
        st.write(formatted_context)

if __name__ == "__main__":
    main() 