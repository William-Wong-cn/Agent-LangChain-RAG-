from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def build_vectorstore(doc_path, embeddings):#RAG
    loader = TextLoader(doc_path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    documents = splitter.split_documents(docs)

    return FAISS.from_documents(documents, embeddings)

def rag_search(vectorstore, query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "【本地知识库】未找到相关内容"
    return "\n".join([doc.page_content for doc in results])
