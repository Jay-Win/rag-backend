from pathlib import Path
from typing import Callable, Dict, List
from langchain_core.documents import Document

# Loaders (only those you actually enable via config will be used)
from loaders.pdf_loader import load_pdf
from loaders.docx_loader import load_docx
from loaders.pptx_loader import load_pptx
from loaders.html_loader import load_html
from loaders.csv_loader import load_csv
from loaders.md_loader import load_md
from loaders.txt_loader import load_txt
from loaders.rtf_loader import load_rtf
from loaders.eml_loader import load_eml
from loaders.excel_loader import load_excel
from loaders.image_ocr_loader import load_image_ocr
from loaders.video_loader import load_mp4
from loaders.doc_loader import load_doc
from loaders.json_loader import load_json


def build_loaders_map(loaders_cfg: dict) -> Dict[str, Callable[[Path], List[Document]]]:
    """
    Build extension -> loader map based on config flags.
    NOTE: Path.suffix is lowercased by the caller; we still define keys lowercase.
    """
    loaders_map: Dict[str, Callable[[Path], List[Document]]] = {}

    # Core you said you want
    if loaders_cfg.get("txt", True):
        loaders_map.update({
            ".txt": load_txt,
            ".text": load_txt,   # alias
            ".log": load_txt,    # alias
        })
    if loaders_cfg.get("pdf", True):
        loaders_map[".pdf"] = load_pdf
    if loaders_cfg.get("docx", True):
        loaders_map[".docx"] = load_docx
    if loaders_cfg.get("json", True):
        loaders_map[".json"] = load_json


    # Optional (kept, but default False—enable in config.yaml if needed)
    if loaders_cfg.get("doc", True):
        loaders_map[".doc"] = load_doc
    if loaders_cfg.get("pptx", False):
        loaders_map[".pptx"] = load_pptx
    if loaders_cfg.get("html", False):
        loaders_map.update({".html": load_html, ".htm": load_html})
    if loaders_cfg.get("csv", False):
        loaders_map[".csv"] = load_csv
    if loaders_cfg.get("md", True):
        loaders_map[".md"] = load_md
    if loaders_cfg.get("rtf", False):
        loaders_map[".rtf"] = load_rtf
    if loaders_cfg.get("eml", False):
        loaders_map[".eml"] = load_eml
    if loaders_cfg.get("excel", False):
        loaders_map.update({".xlsx": load_excel, ".xls": load_excel})
    if loaders_cfg.get("images_ocr", False):
        loaders_map.update({
            ".png": load_image_ocr, ".jpg": load_image_ocr, ".jpeg": load_image_ocr, ".webp": load_image_ocr
        })
    if loaders_cfg.get("video", False):
        loaders_map[".mp4"] = load_mp4

    return loaders_map

def supported_exts(loaders_cfg: dict) -> List[str]:
    """Optional: quick way to print which extensions will be scanned."""
    return sorted(build_loaders_map(loaders_cfg).keys())
