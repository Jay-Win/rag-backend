from pathlib import Path
from typing import List
from PIL import Image
import pytesseract
from langchain_core.documents import Document

def load_image_ocr(path: Path) -> List[Document]:
    try:
        img = Image.open(path)
        # Use German+English; if 'deu' missing, this will fail visibly
        txt = pytesseract.image_to_string(img, lang="deu+eng")
    except Exception as e:
        print(f"⚠️ OCR failed for {path.name}: {e}")
        return []  # <-- do NOT store failure text in the DB

    txt = (txt or "").strip()
    if not txt:
        print(f"ℹ️ OCR found no text in {path.name}; skipping.")
        return []

    return [Document(
        page_content=f"[DOC_NAME: {path.name}]\n{txt}",
        metadata={
            "type": "image",
            "doc_name": path.name,
            "source": str(path),
        },
    )]
