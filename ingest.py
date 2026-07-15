import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_docs():
    docs = []
    folder = "data/docs"

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path).load())

        elif file.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())

        else:
            print(f"Skipping unsupported file: {file}")

    return docs

def ingest():
    docs = load_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="db"
    )

    db.persist()
    print("Ingestion complete. Vector DB saved to /db.")

if __name__ == "__main__":
    ingest()
