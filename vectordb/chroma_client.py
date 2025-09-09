from langchain_chroma import Chroma
from embeddings.get_embedding_function import get_embedding_function

def get_chroma(persist_directory: str) -> Chroma:
    return Chroma(persist_directory=persist_directory, embedding_function=get_embedding_function())
