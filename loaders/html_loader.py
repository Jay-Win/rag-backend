from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document

def load_html(path: Path) -> List[Document]:
    loader = UnstructuredHTMLLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata["type"] = "html"
        d.metadata["source"] = str(path)
        d.metadata["doc_name"] = path.name
    return docs
