from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import faiss
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
import os
from configure import api_key

# Path to the directory containing PDFs
pdf_directory = './PDF Files'

# Output file for storing the FAISS index and documents
embeddings_file = "./embeddings.pkl"

# Load all PDF files from the directory
loader = DirectoryLoader(pdf_directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Set the environment variable for Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = embeddings_api_key

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings()

# Create a FAISS vector store from the documents
db = faiss.FAISS.from_documents(docs, embeddings)

# Save the FAISS index and documents to a file for future use
with open(embeddings_file, "wb") as f:
    pickle.dump(db, f)

print("Embeddings processed and saved.")
