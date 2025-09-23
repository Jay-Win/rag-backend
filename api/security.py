# api/security.py
import os
from typing import Optional
from fastapi import Header, HTTPException

API_KEY = os.getenv("OPAL_API_KEY", "my-secret-key")

def check_key(x_api_key: Optional[str] = Header(None)) -> None:
    """Raise 401 if header doesn't match."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
