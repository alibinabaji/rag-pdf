import os
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)
CORS(app)

pdf_file = ""
API_BASE_URL = ""
API_KEY = ""

def load_and_process_pdf(pdf_file):
    loader = PyPDFLoader(file_path=pdf_file)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    split_documents = text_splitter.split_documents(documents)
    return split_documents

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def setup_vectorstore():
    documents = load_and_process_pdf(pdf_file)
    
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    
    vectorstore.save_local("faiss_index")
    return vectorstore

llm = ChatOpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
    model="gpt-4o-mini",
    temperature=0
)

template = """
متن: {context}

سوال: {question}

پاسخ:"""

prompt = ChatPromptTemplate.from_template(template)

def setup_rag_chain():
    try:
        vectorstore = FAISS.load_local("faiss_index", embeddings)
        print("Loaded existing FAISS index")
    except:
        print("Creating new FAISS index")
        vectorstore = setup_vectorstore()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, vectorstore

def query_rag(query: str, rag_chain, vectorstore):
    """
    Query the RAG system with a question
    """
    answer = rag_chain.invoke(query)
    
    docs = vectorstore.similarity_search(query, k=3)
    
    return {
        "answer": answer,
        "source_documents": docs
    }

print("Initializing RAG system...")
rag_chain, vectorstore = setup_rag_chain()
print("RAG system initialized!")

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('user_message')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
            
        result = query_rag(user_message, rag_chain, vectorstore)
        
        response = {
            'response': result['answer'],
            'sources': [
                {
                    'page': doc.metadata.get('page', 'N/A'),
                    'content': doc.page_content[:200]
                }
                for doc in result['source_documents']
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 