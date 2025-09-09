from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document

def load_excel(path: Path) -> List[Document]:
    docs: List[Document] = []
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet).fillna("")
        headers = list(df.columns)
        for idx, row in df.iterrows():
            row_txt = " | ".join(f"{h}: {row[h]}" for h in headers)
            content = f"[SHEET] {sheet}\n{row_txt}"
            docs.append(Document(
                page_content=content,
                metadata={"type":"excel","source":str(path),"doc_name":path.name,"sheet":sheet,"row":int(idx)}
            ))
    return docs
