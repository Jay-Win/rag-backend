# query_data.py — permissive retrieval, strict answerability

import argparse
import re
import string
from typing import Optional, Dict, Any, List, Tuple

from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

# -----------------------
# Config
# -----------------------

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a careful assistant that must ONLY use the provided context.

CONTEXT:
{context}

STRONG ANCHORS (must appear in the context to answer):
{strong_anchors}

OTHER ANCHORS (nice-to-have):
{soft_anchors}

TASK:
1) Try to answer the user's QUESTION strictly using the CONTEXT.
2) If the CONTEXT does not clearly contain information that answers the QUESTION,
   or at least one STRONG ANCHOR is absent from the CONTEXT and source names,
   respond with exactly: UNKNOWN

QUESTION:
{question}
"""


STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","without","by",
    "is","are","was","were","be","been","being","at","as","from","that","this",
    "it","its","into","over","about","how","do","does","did","can","could",
    "what","when","where","why","which","who","whom","whose",
}

# retrieval score floor (still overridable by changing constant)
SCORE_MIN = 0.30

# -----------------------
# Helpers
# -----------------------

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")

def _extract_filename(q: str) -> Optional[str]:
    m = re.search(r"'([^']+\.\w+)'|\"([^\"]+\.\w+)\"|(\S+\.\w+)", q)
    if not m:
        return None
    return next(g for g in m.groups() if g)

def _infer_filter_from_text(q: str) -> Dict[str, Any]:
    ql = q.lower()
    fname = _extract_filename(ql)
    if fname:
        return {"doc_name": {"$eq": fname}}
    if any(k in ql for k in ["image", "photo", "picture", "screenshot"]):
        return {"type": {"$eq": "image"}}
    if any(k in ql for k in ["video", "clip", "recording", "mp4"]):
        return {"type": {"$eq": "video"}}
    return {}

def _build_meta_filter(args: argparse.Namespace, question: str) -> Dict[str, Any]:
    if getattr(args, "file", None):
        return {"doc_name": {"$eq": args.file}}
    if getattr(args, "type", None):
        return {"type": {"$eq": args.type.lower()}}
    return _infer_filter_from_text(question)

def _loc_str(md: dict) -> str:
    keys = ["page", "slide", "sheet", "row", "section", "element_id", "time_start", "time_end"]
    bits: List[str] = []
    for k in keys:
        v = md.get(k)
        if v not in (None, "", "none"):
            if k in ("time_start", "time_end"):
                continue
            bits.append(f"{k}={v}")
    ts, te = md.get("time_start"), md.get("time_end")
    if ts is not None or te is not None:
        def _fmt(t):
            try:
                t = float(t)
                m, s = divmod(int(round(t)), 60)
                return f"{m:02d}:{s:02d}"
            except Exception:
                return str(t)
        if ts is not None and te is not None:
            bits.append(f"[{_fmt(ts)}–{_fmt(te)}]")
        elif ts is not None:
            bits.append(f"[{_fmt(ts)}→]")
    return " ".join(bits)

def _format_sources(docs: List) -> str:
    lines = []
    for d in docs:
        md = d.metadata or {}
        name = md.get("doc_name") or md.get("source") or "unknown"
        typ = md.get("type") or "doc"
        loc = _loc_str(md)
        lines.append(f"• {name} ({typ})" + (f" — {loc}" if loc else ""))
    return "\n".join(lines)

def _truncate_context(chunks: List, max_chars: int) -> List:
    if max_chars <= 0:
        return chunks
    total = 0
    out = []
    for d in chunks:
        c = len(d.page_content or "")
        if total + c > max_chars and out:
            break
        out.append(d)
        total += c
    return out

def _dedup_by_source(docs: List, per_source_limit: int = 2) -> List:
    from collections import defaultdict
    bucket = defaultdict(list)
    for d in docs:
        src = (d.metadata or {}).get("source", "unknown")
        if len(bucket[src]) < per_source_limit:
            bucket[src].append(d)
    out: List = []
    for v in bucket.values():
        out.extend(v)
    return out

# -------- Anchor extraction & answerability checks --------

def _bonus_rank(candidates: List[Tuple[Any, float]], anchors: List[str]) -> List[Tuple[Any, float]]:
    """
    Small score bonus if a candidate contains any anchor in text or name.
    Keeps ordering stable otherwise.
    """
    bumped: List[Tuple[Any, float]] = []
    for d, s in candidates:
        txt = (d.page_content or "").lower()
        md = d.metadata or {}
        name = (md.get("doc_name") or md.get("source") or "").lower()
        if any(a in txt for a in anchors) or any(a in name for a in anchors):
            s = min(1.0, s + 0.05)
        bumped.append((d, s))
    bumped.sort(key=lambda x: x[1], reverse=True)
    return bumped

# -------- Anchor extraction & answerability checks --------

def _extract_anchors(question: str) -> tuple[list[str], list[str]]:
    """
    Return (strong, soft) anchors:
      - strong: quoted phrases + Proper Nouns (e.g., Monopoly, Ticket to Ride)
      - soft: other long tokens (len>=4) not in STOPWORDS
    All lowercase, unique, order-preserving.
    """
    # quoted phrases
    quoted_pairs = re.findall(r'"([^"]+)"|\'([^\']+)\'', question)
    quoted = [a or b for (a, b) in quoted_pairs]

    # Proper Nouns (very simple heuristic; keep original casing to detect)
    proper_raw = re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", question)

    # long-ish tokens (lower)
    toks = [
        t for t in re.findall(r"\w+", question.lower())
        if len(t) >= 4 and t not in STOPWORDS
    ]

    strong: list[str] = []
    soft: list[str] = []
    seen = set()

    def _push(dst: list[str], x: str):
        x = x.strip().lower()
        if x and x not in seen:
            seen.add(x)
            dst.append(x)

    for s in quoted:
        _push(strong, s)
    for s in proper_raw:
        _push(strong, s)
    for t in toks:
        if t not in strong:
            _push(soft, t)

    return strong, soft


def _context_has_any(docs: list, needles: list[str]) -> bool:
    """True if any needle appears in concatenated text OR in any doc_name/source."""
    if not needles:
        return False
    ctx_text = " ".join((d.page_content or "").lower() for d in docs)
    if any(a in ctx_text for a in needles):
        return True
    for d in docs:
        md = d.metadata or {}
        name = (md.get("doc_name") or md.get("source") or "").lower()
        if any(a in name for a in needles):
            return True
    return False


def _bonus_rank(candidates: list[tuple], anchors_all: list[str]) -> list[tuple]:
    """Small score bonus if a candidate contains any anchor (in text or name)."""
    bumped = []
    for d, s in candidates:
        txt = (d.page_content or "").lower()
        md = d.metadata or {}
        name = (md.get("doc_name") or md.get("source") or "").lower()
        if any(a in txt for a in anchors_all) or any(a in name for a in anchors_all):
            s = min(1.0, s + 0.05)
        bumped.append((d, s))
    bumped.sort(key=lambda x: x[1], reverse=True)
    return bumped


# -----------------------
# Main RAG
# -----------------------

def query_rag(query_text: str, args: argparse.Namespace) -> str:
    """
    Permissive retrieval (can return related chunks),
    but strict answerability:
      - LLM receives STRONG/SOFT ANCHORS and is told to output UNKNOWN if unsupported.
      - Post-LLM guard forces UNKNOWN when STRONG anchors are absent across context/names.
    Keeps: score floor, strict file retry, guardrail, per-source dedupe, truncation, CLI prints.
    """
    import os  # keep local like before

    # --- Vector DB ---
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # --- Optional metadata filter ---
    meta_filter = _build_meta_filter(args, query_text)

    # ---- Retrieval with relevance scores ----
    fetch_k = max(args.fetch_k, args.k)
    docs_with_scores: List[Tuple[Any, float]] = db.similarity_search_with_relevance_scores(
        query=query_text,
        k=fetch_k,
        filter=meta_filter if meta_filter else None,  # LangChain-Chroma expects `filter=`
    )

    # 1) drop weak matches by score
    candidates: List[Tuple[Any, float]] = [(d, s) for (d, s) in docs_with_scores if s >= SCORE_MIN]

    # 2) gentle anchor-aware re-ranking (no hard filter)
    strong, soft = _extract_anchors(query_text)
    anchors_all = strong + soft
    if anchors_all:
        candidates = _bonus_rank(candidates, anchors_all)

    # 3) finalize top-k
    candidates = candidates[: args.k]
    docs = [d for (d, _) in candidates]

    # ---- Strict retry when a specific file was requested but nothing survived ----
    requested_file = (args.file or _extract_filename(query_text) or "").lower().strip()
    if requested_file and not docs:
        broad = db.similarity_search_with_relevance_scores(query_text, k=fetch_k)
        filtered: List[Tuple[Any, float]] = []
        for d, s in broad:
            md = d.metadata or {}
            dn = (md.get("doc_name") or "").lower().strip()
            bn = os.path.basename(md.get("source") or "").lower().strip()
            if dn == requested_file or bn == requested_file:
                filtered.append((d, s))
        filtered.sort(key=lambda x: x[1], reverse=True)
        docs = [d for (d, _) in filtered[: args.k]]

    # ---- Guardrail (retain original semantics) ----
    if "speed die" not in query_text.lower():
        tmp = []
        for d in docs:
            t = (d.page_content or "").lower()
            if "speed die" in t or "mr. monopoly" in t:
                continue
            tmp.append(d)
        if len(tmp) >= max(3, args.k // 2):
            docs = tmp

    # ---- Dedupe & truncate ----
    docs = _dedup_by_source(docs, per_source_limit=args.per_source_limit)
    docs = _truncate_context(docs, max_chars=args.max_context_chars)

    # ---- Early UNKNOWN if nothing survived ----
    if not docs:
        print("---- Retrieved snippets ----")
        print("(no confident matches after retrieval)")
        print("----------------------------")
        print("Response: UNKNOWN")
        print("----------------------------")
        print("Sources:")
        return "UNKNOWN"

    # ---- Debug snippets ----
    print("---- Retrieved snippets ----")
    if getattr(args, "show_snippets", False):
        for i, d in enumerate(docs[: args.k], 1):
            snippet = (d.page_content or "")[: getattr(args, "snippet_chars", 220)].replace("\n", " ")
            print(f"[{i}] {d.metadata.get('doc_name') or d.metadata.get('source')} → {snippet}...")
    else:
        print("(use --show-snippets to see text previews)")
    print("----------------------------")

    # ---- Build prompt (include STRONG/SOFT anchors) ----
    context_text = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text,
        strong_anchors=", ".join(strong) if strong else "(none)",
        soft_anchors=", ".join(soft) if soft else "(none)",
        question=query_text,
    )

    # ---- LLM call ----
    llm = ChatOllama(model=args.model)
    msg = llm.invoke(prompt)
    answer = getattr(msg, "content", str(msg)).strip()

    # ---- Post-LLM answerability guard ----
    # If there are STRONG anchors (e.g., "Monopoly") and they do not appear
    # in the context text or in any source/doc_name, force UNKNOWN.
    if answer.upper() != "UNKNOWN":
        if strong and not _context_has_any(docs, strong):
            answer = "UNKNOWN"

    print("Response:", answer)
    print("----------------------------")
    print("Sources:")
    print(_format_sources(docs[: args.k]))
    return answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query_text", type=str, help="Your question")
    # Retrieval tuning
    p.add_argument("--k", type=int, default=12, help="Final top-k returned to context")
    p.add_argument("--fetch-k", type=int, default=48, help="Candidates fetched before MMR")
    p.add_argument("--per-source-limit", type=int, default=2, help="Max chunks per source file")
    # Filters
    p.add_argument("--type", choices=[
        "pdf", "docx", "doc", "image", "video", "html", "csv", "md", "txt", "rtf", "eml", "excel"
    ], help="Filter by document type")
    p.add_argument("--file", help="Filter to a specific file name (exact match, e.g. 'monopoly.pdf')")
    # Display / speed
    p.add_argument("--show-snippets", action="store_true", help="Print short previews of retrieved chunks")
    p.add_argument("--snippet-chars", type=int, default=220, help="Chars per snippet when --show-snippets")
    p.add_argument("--max-context-chars", type=int, default=12000, help="Cap total context length for speed")
    # LLM model
    p.add_argument("--model", default="mistral", help="Ollama chat model name")
    args = p.parse_args()

    query_rag(args.query_text, args)


if __name__ == "__main__":
    main()
