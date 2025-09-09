from pathlib import Path
from typing import List
from faster_whisper import WhisperModel
from langchain_core.documents import Document

def load_mp4(path: Path) -> List[Document]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(f"⚠️ Skipping {path.name}: faster-whisper not installed")
        return []

    model = WhisperModel("tiny")  # downloads on first run

    try:
        segments, _info = model.transcribe(str(path))
    except Exception as e:
        print(f"⚠️ Transcription failed for {path.name}: {e}")
        return []

    docs: List[Document] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "type": "video",
                "source": str(path),
                "doc_name": path.name,
                "start": float(seg.start),
                "end": float(seg.end),
            }
        ))
    if not docs:
        print(f"ℹ️ No speech recognized in {path.name}; skipping.")
    return docs
