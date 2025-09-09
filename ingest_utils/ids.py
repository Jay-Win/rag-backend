import hashlib
from collections import defaultdict
from langchain_core.documents import Document

def build_locator(md: dict) -> str:
    parts = []
    for key in ("page", "slide", "sheet", "row", "section", "element_id"):
        if key in md and md[key] not in (None, ""):
            parts.append(f"{key}={md[key]}")
    return ";".join(parts) if parts else ""

def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]

def assign_ids(chunks: list[Document]) -> list[Document]:
    counters = defaultdict(int)  # (source, locator) -> next index
    for d in chunks:
        source = d.metadata.get("source", "unknown")
        locator = build_locator(d.metadata)
        idx = counters[(source, locator)]
        h = short_hash((d.page_content or "") + locator)
        d.metadata["id"] = f"{source}:{locator}:{idx}:{h}"
        counters[(source, locator)] += 1
    return chunks
