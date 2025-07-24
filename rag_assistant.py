import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()


embeddings = OpenAIEmbeddings()  

# === Load all .txt/.md documents from a folder ===
def load_docs_from_folder(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt") or file.endswith(".md"):
            try:
                docs.extend(TextLoader(os.path.join(folder_path, file), encoding="utf-8").load())
            except Exception as e:
                print(f"❌ Failed to load {file}: {e}")
    return docs

# === Create FAISS vector store from docs ===
def create_vectorstore(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# === Create RAG chain ===
def create_rag_chain(vectordb):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# === Final RAG Assistant callable ===
def ask_question(query, file_path="./knowledge_base"):
    try:
        # 1. Load documents from folder
        docs = load_docs_from_folder(file_path)
        if not docs:
            return "❌ No documents found in the specified knowledge base."

        # 2. Create vector store
        vectordb = create_vectorstore(docs)
        if not vectordb:
            return "❌ Failed to create or load vector store."

        # 3. Create RAG chain
        qa_chain = create_rag_chain(vectordb)
        if not qa_chain:
            return "❌ Failed to initialize RAG chain."

        # 4. Run query safely
        result = qa_chain.run(query)
        return result if result else "❌ No answer could be generated from the documents."

    except Exception as e:
        return f"❌ Error in ask_question(): {e}"
