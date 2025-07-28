from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
DATA_PATH = './data'
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(DATA_PATH)
print("Length of PDF pages: ", len(documents))

# Create chunks
def create_chunks(data):
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    chunks = text_spliter.split_documents(data)
    return chunks

text_chunks=create_chunks(documents)
print("Length of Text Chunks: ", len(text_chunks))

# Create vector embeddings
def create_vector_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = create_vector_embeddings()

# Store embeddings in vector database
DB_FAISS = './vectordatabase'
db = FAISS.from_documents(text_chunks, embeddings)
db.save_local(DB_FAISS)