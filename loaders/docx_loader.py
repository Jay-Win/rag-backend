from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_core.documents import Document

def load_docx(path: Path) -> List[Document]:
    # mode="elements" returns many smaller Documents (titles, paragraphs, lists)
    loader = UnstructuredWordDocumentLoader(str(path), mode="elements")
    docs = loader.load()
    for d in docs:
        d.metadata["type"] = "docx"
        d.metadata["source"] = str(path)
        d.metadata["doc_name"] = path.name
    return docs
