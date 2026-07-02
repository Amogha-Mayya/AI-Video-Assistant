from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

COLLECTION_NAME = "meeting_transcript"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    # No persist_directory = in-memory only.
    # Faster build, no disk writes, works correctly on Streamlit Cloud
    # (cloud filesystem resets between sessions anyway so persisting was pointless)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def get_retriever(vector_store: Chroma, k: int = 3):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
