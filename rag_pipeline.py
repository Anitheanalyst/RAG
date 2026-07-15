from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def rag_answer(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    vectorstore = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    docs = vectorstore.similarity_search(query, k=4)
    content = "\n\n".join([d.page_content for d in docs])

    return content

