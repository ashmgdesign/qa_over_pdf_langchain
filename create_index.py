from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = api_key

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

def extract_text_from_single_pdf_file(pdf_path):
    pdf_loader = PyMuPDFLoader( 
        pdf_path) 
    
    pdf_data = pdf_loader.load()  # Load PDF files 
    return pdf_data

def extract_text_from_multiple_pdf_files(directory_path):
    pdf_loader = PyPDFDirectoryLoader( 
        directory_path, silent_errors=True) 
    
    pdf_data = pdf_loader.load()  # Load PDF files 
    return pdf_data

data = extract_text_from_single_pdf_file("./data/LCViva 29.3 Copy.pdf") # Loading single PDF

# data = extract_text_from_multiple_pdf_files("./data") # Use this if you plan to load multiple PDFs at once

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800, # You can play around with this parameter to adjust the length of each chunk
    chunk_overlap  = 20,
    length_function = len,
)

new_chunks = None
for obj in data:
    if new_chunks is None:
        new_chunks = text_splitter.create_documents([obj.page_content], metadatas=[obj.metadata])
    else:
        new_chunks = new_chunks + text_splitter.create_documents([obj.page_content], metadatas=[obj.metadata])

db = FAISS.from_documents(new_chunks, embeddings)

db.save_local("ash_local_FAISS_index") # Name of index you want to store locally. We'll use this in the other script to interact with your document.