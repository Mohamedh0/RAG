# Libraries
import os
import warnings
import logging
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Configure warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('📄 AI Chatbot with PDF Support')

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file (optional)", type=["pdf"])

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'file_processed' not in st.session_state:
    st.session_state.file_processed = False
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def create_vectorstore(file_content):
    """Processes the uploaded PDF and creates a FAISS vector store."""
    try:
        # Use a temporary file path in the system's temp directory
        temp_pdf_path = os.path.join(os.path.dirname(__file__), "temp.pdf")
        
        with open(temp_pdf_path, "wb") as f:
            f.write(file_content)
        
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
        # Ensure the temporary file is removed
        os.remove(temp_pdf_path)

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"Failed to process PDF: {str(e)}")
        return None

# Process uploaded file
if uploaded_file and not st.session_state.file_processed:
    with st.spinner("Processing PDF..."):
        file_content = uploaded_file.read()
        st.session_state.vectorstore = create_vectorstore(file_content)
        if st.session_state.vectorstore:
            st.session_state.file_processed = True
            st.success("✅ PDF processed successfully! You can now ask questions.")

# Chat input
prompt = st.chat_input('💬 Type your question here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Use the latest supported model
    try:
        groq_chat = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),  
            model_name="llama3-8b-8192",
            temperature=0.1
        )
    except Exception as e:
        st.error(f"Error initializing Groq API: {str(e)}")
        st.stop()
    
    try:
        if st.session_state.vectorstore:  # PDF-based Q&A mode
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 4})

            # Retrieve context documents
            context_docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in context_docs])

            # Prepare the prompt with context
            full_prompt = f"""You are an expert AI assistant with access to a document. 
            Provide a detailed, accurate answer based on the document content.

            Context: {context}
            User Query: {prompt}"""

            # Generate response
            messages = [
                {"role": "system", "content": "You are an expert document assistant."},
                {"role": "user", "content": full_prompt}
            ]
            response = groq_chat.invoke(messages).content

        else:  # General chatbot mode
            # Prepare the prompt with conversation history
            history = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in st.session_state.messages[-5:]  # Limit history to last 5 messages
            ])

            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": f"Conversation History:\n{history}\n\nLatest Query: {prompt}"}
            ]
            response = groq_chat.invoke(messages).content

        # Display and store response
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        response = "I encountered an error. Please try again."
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

# Reset button
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.session_state.vectorstore = None
    st.session_state.file_processed = False
    st.session_state.memory.clear()
    st.rerun()