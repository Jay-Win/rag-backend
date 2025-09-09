from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredRTFLoader
from langchain_core.documents import Document

def load_rtf(path: Path) -> List[Document]:
    docs = UnstructuredRTFLoader(str(path)).load()
    for d in docs:
        d.metadata.update({"type":"rtf","source":str(path),"doc_name":path.name})
        d.metadata["doc_name"] = path.name
    return docs
