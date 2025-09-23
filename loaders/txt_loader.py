from langchain_core.documents import Document

def load_txt(path):
    """
    Robust TXT loader:
    - tries several common Windows encodings
    - last resort: UTF-8 with errors='replace'
    - trims only for emptiness check; preserves original content for DB
    """
    tried = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = path.read_text(encoding=enc, errors="strict")
            # consider content 'empty' if no visible characters at all
            if raw and raw.strip():
                return [Document(
                    page_content=raw,
                    metadata={
                        "source": str(path.resolve()),
                        "doc_name": path.name,
                        "type": "txt",
                    },
                )]
            else:
                # allow logging upstream that this file decoded but is empty
                return []
        except Exception as e:
            tried.append(f"{enc}: {e.__class__.__name__}: {e}")

    # last resort: decode with replacement to salvage something
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        if raw and raw.strip():
            return [Document(
                page_content=raw,
                metadata={
                    "source": str(path.resolve()),
                    "doc_name": path.name,
                    "type": "txt",
                },
            )]
    except Exception:
        pass

    # Let ingest know we failed decoding
    print(f"⚠️ TXT decode failed for {path.name}. Tried: {', '.join(tried)}")
    return []
