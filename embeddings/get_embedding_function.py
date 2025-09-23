# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings


# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default", region_name="us-east-1"
#     )
#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings


import os
from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # <-- read env
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")    # pick yours

    return OllamaEmbeddings(
        model=model,
        base_url=base_url,  # <-- IMPORTANT
    )