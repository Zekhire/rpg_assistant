from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from settings import CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_DIR, EMBEDDING_MODEL, PDF_PATH


def ingest_pdf():
    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    # 2. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # 4. Store in ChromaDB
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("PDF ingested into ChromaDB")

if __name__ == "__main__":
    ingest_pdf()