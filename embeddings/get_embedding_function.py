# from langchain_community.embeddings.ollama import OllamaEmbeddings
# from langchain_community.embeddings.bedrock import BedrockEmbeddings


# def get_embedding_function():
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default", region_name="us-east-1"
#     )
#     # embeddings = OllamaEmbeddings(model="nomic-embed-text")
#     return embeddings


from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # Uses your local Ollama at http://localhost:11434
    # If your Ollama runs elsewhere, add base_url="http://HOST:11434"
    return OllamaEmbeddings(model="nomic-embed-text")
