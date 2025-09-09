# OPAL RAG Tutorial

This project is a **Retrieval-Augmented Generation (RAG)** pipeline built on top of:
- [LangChain](https://python.langchain.com)
- [Ollama](https://ollama.com)
- [Chroma](https://www.trychroma.com) as vector database
- OCR (via [Tesseract](https://github.com/tesseract-ocr/tesseract)) for images
- ffmpeg for video frame analysis

It ingests documents (PDF, Word, Excel, text, images, video, …) into a vector database and lets you query them with natural language.

---

## 1. Ingesting Documents

```bash
py ingest.py [--reset] [--rescan]



# Fresh start: clear DB and ingest everything
py ingest.py --reset

# Normal incremental update: add new/changed files only
py ingest.py

# Force rescan all files (ignore manifest cache)
py ingest.py --rescan



py query_data.py "your question" [options]

# Ask normally (whole corpus)
py query_data.py "Summarize Cohesity's 'One platform' benefits"

# Filter by file
py query_data.py "What are the backup features?" --file "4-strategien-fuer-ein-datengetriebenes-unternehmen.pdf"

# Filter by type (images only)
py query_data.py "What text is written on the coin image?" --type image --show-snippets

# Fetch more, keep context small
py query_data.py "How do you win in Monopoly?" --fetch-k 80 --k 12 --max-context-chars 10000

# Use another Ollama model
py query_data.py "Give me a summary of the OPAL software concept" --model llama3
