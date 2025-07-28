import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup LLM
HF_TOKEN = os.environ.get('HF_TOKEN')
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

def load_llm(HUGGINGFACE_REPO_ID):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                    "max_length":"512"}
    )
    return llm

# Connect LLM with FAISS and create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you do not know the answer, just say that you do not know, do not try to make up an answer. 
Do not provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

# Load Vector Database
DB_FAISS_PATH = './vectordatabase'
embeddings_model = HuggingFaceEmbeddings(model_name = model_name)
db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])