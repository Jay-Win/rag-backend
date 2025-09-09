from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def load_txt(path: Path) -> List[Document]:
    docs = TextLoader(str(path), encoding="utf-8").load()
    for d in docs:
        d.metadata.update({"type":"txt","source":str(path),"doc_name":path.name})
        d.metadata["doc_name"] = path.name
    return docs
