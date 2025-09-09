from pathlib import Path
from typing import List
from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document

def load_csv(path: Path) -> List[Document]:
    # CSVLoader turns each row into a Document with page_content as row text
    loader = CSVLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for idx, d in enumerate(docs):
        d.metadata["type"] = "csv"
        d.metadata["source"] = str(path)
        d.metadata.setdefault("row", idx)
        d.metadata["doc_name"] = path.name
    return docs
