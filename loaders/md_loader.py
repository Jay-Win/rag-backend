from pathlib import Path
from typing import List
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

def load_md(path: Path) -> List[Document]:
    docs = UnstructuredMarkdownLoader(str(path)).load()
    for d in docs:
        d.metadata.update({"type":"md","source":str(path),"doc_name":path.name})
        d.metadata["doc_name"] = path.name
    return docs
