import json
from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load_json(path: Path) -> List[Document]:
    """
    Load a JSON file and convert it into a Document.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert JSON into a nice string (flatten nested objects)
        text = json.dumps(data, indent=2, ensure_ascii=False)

        return [
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "doc_name": path.name,
                    "type": "json"
                }
            )
        ]
    except Exception as e:
        return [
            Document(
                page_content=f"[ERROR loading JSON: {e}]",
                metadata={
                    "source": str(path),
                    "doc_name": path.name,
                    "type": "json"
                }
            )
        ]
