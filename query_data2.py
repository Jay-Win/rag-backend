# query_data2.py

import os
import argparse
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from ollama import Client

# ---- Config (env overridable) ----
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://h01.m5.jay-win.de:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

PROMPT_TEMPLATE = """
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer is not clearly in the context, respond with exactly: UNKNOWN

CONTEXT:
{context}

QUESTION:
{question}
"""

def get_embedding_function():
    # Embeddings use the same Ollama host
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)

def _make_filter(args: argparse.Namespace) -> Dict[str, Any] | None:
    filt: Dict[str, Any] = {}
    if args.file:
        filt["doc_name"] = args.file
    if args.type:
        filt["type"] = args.type
    return filt or None

def _client_side_filter(docs, args: argparse.Namespace):
    # Extra safeguard in case server-side filter is too lax / unsupported
    target = (args.file or "").strip()
    typ = (args.type or "").strip()
    if not target and not typ:
        return docs
    out = []
    for d in docs:
        md = d.metadata or {}
        ok = True
        if target:
            dn = (md.get("doc_name") or "").strip()
            src = (md.get("source") or "").strip()
            ok = ok and (dn == target or target in src)
        if typ:
            ok = ok and (md.get("type") == typ)
        if ok:
            out.append(d)
    return out

def query_rag(query_text: str, args: argparse.Namespace) -> str:
    # Connect to Chroma
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=get_embedding_function())

    # Build optional metadata filter
    meta_filter = _make_filter(args)

    # Retrieve (fetch a bit more if filtering)
    fetch_k = max(args.k, 5)
    if meta_filter:
        fetch_k = max(args.k * 3, 15)

    docs = db.similarity_search(query_text, k=fetch_k, filter=meta_filter)
    docs = _client_side_filter(docs, args)
    docs = docs[:args.k]

    if not docs:
        print("Response: UNKNOWN")
        return "UNKNOWN"

    # Show retrieved chunks for logs/raw
    print("---- Retrieved chunks ----")
    for i, d in enumerate(docs, 1):
        snippet = (d.page_content[:300] + "…") if len(d.page_content) > 300 else d.page_content
        print(f"[{i}] {d.metadata.get('doc_name') or d.metadata.get('source')} -> {snippet}")
    print("--------------------------")

    # Build context & prompt
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context, question=query_text
    )
    user_prompt = str(prompt)

    # Chat client (same Ollama host as embeddings)
    client = Client(host=OLLAMA_HOST)

    try:
        resp = client.chat(
            model=args.model,
            messages=[{"role": "user", "content": user_prompt}],
            options={"temperature": 0},
            stream=False,
        )
        answer = (resp.get("message") or {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Empty response from model")
    except Exception as e:
        print("Response: UNKNOWN")
        print("Sources:")
        print(f"- ERROR: {type(e).__name__}: {e}")
        return "UNKNOWN"

    print("Response:", answer)
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Your question")
    parser.add_argument("--k", type=int, default=5, help="How many chunks to retrieve")
    parser.add_argument("--model", default="mistral", help="Ollama model to use")
    parser.add_argument("--file", default="", help="Restrict to metadata.doc_name")
    parser.add_argument("--type", default="", help="Restrict to metadata.type")
    args = parser.parse_args()

    query_rag(args.query_text, args)

if __name__ == "__main__":
    main()
