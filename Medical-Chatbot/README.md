# 🏥 Medical Chatbot

A RAG (Retrieval-Augmented Generation) powered medical chatbot that provides accurate medical information by querying a pre-built knowledge base of medical documents. Built with LangChain, FAISS, and HuggingFace models, deployed via Streamlit for an interactive web interface.

## ✨ Features

- **Medical Q&A**: Ask medical questions and get accurate, context-based responses
- **RAG Architecture**: Combines retrieval from medical documents with LLM generation
- **Source Citations**: Displays source documents for transparency and verification
- **Pre-built Knowledge Base**: Uses FAISS vector database for fast semantic search
- **Conversational Interface**: Clean Streamlit-based chat UI with message history

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **LLM** | HuggingFace Inference API (Mistral-7B-Instruct-v0.3) |
| **Embeddings** | HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | FAISS |
| **Framework** | LangChain |
| **PDF Processing** | PyPDFLoader, DirectoryLoader |

## 📋 Prerequisites

- Python 3.8+
- HuggingFace API Token

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Medical-Chatbot.git
   cd Medical-Chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   HF_TOKEN=your_huggingface_token_here
   ```

4. **Prepare your medical documents** (optional - if creating new vector database):
   - Place PDF files in the `./data` directory
   - Run the memory creation script:
     ```bash
     python create_memory.py
     ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
Medical-Chatbot/
├── app.py                      # Main Streamlit application
├── create_memory.py            # Script to create FAISS vector database from PDFs
├── connect_memory_with_llm.py  # CLI version for testing the QA chain
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── data/                       # Place medical PDF documents here
└── vectordatabase/             # FAISS vector store
    └── index.faiss             # Pre-built vector index
```

## 📖 Usage

### Web Interface (Streamlit)
```bash
streamlit run app.py
```
Then open your browser and navigate to `http://localhost:8501`

### Command Line Interface
```bash
python connect_memory_with_llm.py
```
Enter your medical question when prompted.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Preparation Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│  PDF Files ──▶ PyPDFLoader ──▶ Text Splitter ──▶ Embeddings   |
│   (./data)                     (500 chars)     (MiniLM-L6-v2)   │
│                                                      │          │
│                                              ┌───────▼────────┐ │
│                                              │ FAISS Vector   │ │
│                                              │   Database     │ │
│                                              └───────┬────────┘ │
└──────────────────────────────────────────────────────┼──────────┘
                                                       │
┌──────────────────────────────────────────────────────┼──────────┐
│                    Query Pipeline                    │          │
├──────────────────────────────────────────────────────┼──────────┤
│                                              ┌───────▼────────┐ │
│  User Query ──▶ Embeddings ──▶ Similarity ──▶│   Top-K Docs │ │
│                               Search         │    (k=3)       │ │
│                                              └───────┬────────┘ │
│                                                      │          │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────▼────────┐  │
│  │  Response   │◀───│  Mistral-7B LLM  │◀───│ Context+Query │  │
│  │ + Sources   │    │  (HuggingFace)   │    │    Prompt      │  │
│  └─────────────┘    └──────────────────┘    └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 | Size of text chunks for processing |
| `chunk_overlap` | 50 | Overlap between chunks |
| `temperature` | 0.5 | LLM temperature for response generation |
| `k` | 3 | Number of relevant documents to retrieve |
| `max_length` | 512 | Maximum response length |

## 📚 Adding New Medical Documents

1. Place your PDF files in the `./data` directory
2. Run the vector database creation script:
   ```bash
   python create_memory.py
   ```
3. The script will:
   - Load all PDFs from the data directory
   - Split them into chunks
   - Create embeddings using MiniLM-L6-v2
   - Save the FAISS index to `./vectordatabase`
