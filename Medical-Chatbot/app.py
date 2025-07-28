import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = './vectordatabase'
@st.cache_resource
def get_vector_database(DB_FAISS_PATH = './vectordatabase',model_name='sentence-transformers/all-MiniLM-L6-v2'):
    embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])

def load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                    "max_length":"512"}
    )
    return llm

def main():
    st.title("Medical Chatbot")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input('Write your prompt here:')
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
You are a highly reliable AI medical assistant. Use only the provided context to answer the question.

Context:
{context}

Question:
{question}

Guidelines:
- If the answer is not in the context, say "I don't know" instead of guessing.
- Provide clear, concise, and medically accurate information.
- When applicable, include alternative treatments or additional considerations.

Final Answer:
"""
        
        HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN=os.environ.get("HF_TOKEN")

        try: 
            vectorstore=get_vector_database()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain=RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response=qa_chain.invoke({'query':prompt})

            result=response["result"]
            source_documents=response["source_documents"]
            result_to_show=result+"\nSource Docs:\n"+str(source_documents)
            #response="Hi, I am MediBot!"
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role':'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()