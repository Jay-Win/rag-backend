from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_core.documents import Document

def load_eml(path: Path) -> List[Document]:
    docs = UnstructuredEmailLoader(str(path)).load()
    for d in docs:
        d.metadata.update({"type":"eml","source":str(path),"doc_name":path.name})
        d.metadata["doc_name"] = path.name
    return docs
