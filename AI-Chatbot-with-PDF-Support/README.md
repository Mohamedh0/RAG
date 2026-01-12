# 📄 AI Chatbot with PDF Support

An intelligent conversational AI chatbot built with Streamlit that supports both general conversations and PDF-based question answering using RAG (Retrieval-Augmented Generation).

**[🚀 Live Demo](https://huggingface.co/spaces/Mohamedh0/AI_Chatbot_with_PDF_Support)**

## ✨ Features

- **Dual Mode Operation**:
  - **General Chat Mode**: Engage in natural conversations with the AI assistant
  - **PDF Q&A Mode**: Upload a PDF and ask questions about its content
- **RAG-Powered Responses**: Uses FAISS vector store for semantic document search
- **Conversation Memory**: Maintains chat history for context-aware responses
- **Fast Inference**: Powered by Groq's LLaMA 3 (8B) model for quick responses
- **Semantic Search**: Uses HuggingFace's `all-mpnet-base-v2` embeddings for accurate document retrieval

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **LLM** | Groq API (LLaMA 3-8B-8192) |
| **Embeddings** | HuggingFace Sentence Transformers (`all-mpnet-base-v2`) |
| **Vector Store** | FAISS |
| **Framework** | LangChain |
| **PDF Processing** | PyPDFLoader |

## 📋 Prerequisites

- Python 3.8+
- Groq API Key

## 🚀 Installation

1. **Install dependencies**:
   ```bash
   pip install streamlit langchain langchain-groq langchain-community faiss-cpu pypdf python-dotenv sentence-transformers
   ```

2. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

1. **General Chat**: Simply type your questions in the chat input to have a conversation with the AI
2. **PDF Q&A**: 
   - Upload a PDF file using the file uploader
   - Wait for the document to be processed
   - Ask questions about the document content

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  PDF Processing  │───▶│  Text Splitter  │
└─────────────────┘    │  (PyPDFLoader)   │    │ (1000 chars,    │
                       └──────────────────┘    │  200 overlap)   │
                                               └────────┬────────┘
                                                        │
                       ┌──────────────────┐    ┌────────▼────────┐
                       │   FAISS Vector   │◀───│   Embeddings    │
                       │     Store        │    │ (MPNet-base-v2) │
                       └────────┬─────────┘    └─────────────────┘
                                │
┌─────────────────┐    ┌────────▼─────────┐    ┌─────────────────┐
│    Response     │◀───│   Groq LLaMA 3   │◀───│ Context + Query │
└─────────────────┘    │      (8B)        │    └─────────────────┘
                       └──────────────────┘
```

## 📁 Project Structure

```
AI-Chatbot-with-PDF-Support/
├── app.py              # Main Streamlit application
├── .env                # Environment variables (create this)
└── README.md           # Project documentation
```

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Size of text chunks for processing |
| `chunk_overlap` | 200 | Overlap between chunks |
| `temperature` | 0.1 | LLM temperature for response generation |
| `k` | 4 | Number of relevant documents to retrieve |
