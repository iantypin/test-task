from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


def embed_all():
    loader = DirectoryLoader("datasets/", glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vectorstore/")


if __name__ == "__main__":
    embed_all()
