from pathlib import Path
from typing import Callable, Dict, List
from langchain_core.documents import Document

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

def build_loaders_map(loaders_cfg: dict) -> Dict[str, Callable[[Path], List[Document]]]:
    loaders_map: Dict[str, Callable[[Path], List[Document]]] = {}
    if loaders_cfg.get("pdf", True):   loaders_map.update({".pdf": load_pdf})
    if loaders_cfg.get("docx", True):  loaders_map.update({".docx": load_docx})
    if loaders_cfg.get("pptx", True):  loaders_map.update({".pptx": load_pptx})
    if loaders_cfg.get("html", True):  loaders_map.update({".html": load_html, ".htm": load_html})
    if loaders_cfg.get("csv", True):   loaders_map.update({".csv": load_csv})
    if loaders_cfg.get("md", True):    loaders_map.update({".md": load_md})
    if loaders_cfg.get("txt", True):   loaders_map.update({".txt": load_txt})
    if loaders_cfg.get("rtf", True):   loaders_map.update({".rtf": load_rtf})
    if loaders_cfg.get("eml", True):   loaders_map.update({".eml": load_eml})
    if loaders_cfg.get("excel", True): loaders_map.update({".xlsx": load_excel, ".xls": load_excel})
    if loaders_cfg.get("images_ocr", False):
        loaders_map.update({".png": load_image_ocr, ".jpg": load_image_ocr, ".jpeg": load_image_ocr, ".webp": load_image_ocr})
    if loaders_cfg.get("video", False): loaders_map.update({".mp4": load_mp4})
    if loaders_cfg.get("doc", True):   loaders_map.update({".doc": load_doc})
    return loaders_map
