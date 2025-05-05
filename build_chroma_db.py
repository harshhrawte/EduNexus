# build_chroma_db.py
# ────────────────────────────────────────────────────────────────
# Required Installs:
# pip install -q langchain langchain_groq chromadb python-dotenv unstructured langchain-huggingface

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


from langchain_huggingface import HuggingFaceEmbeddings

# ─── Load environment variables ─────────────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN missing in your .env file!")

# ─── 1) Load all textbooks ─────────────────────────────────────
paths = [
    r"C:\Harsh\Desktop\suvidhacode\books\science 10th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\science 9th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\science 8th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\science 7th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\science 6th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\hist 10th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\hist 9th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\hist 8th.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\hist 7th.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\hist 6 th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\geography 10th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\geo 9th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\geo 8 th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\geo 7th std.pdf",
    r"C:\Harsh\Desktop\suvidhacode\books\geo 6th std.pdf",
]

docs = []
for p in paths:
    if not os.path.isfile(p):
        print(f"⚠️ Skipping missing or invalid file: {p}")
        continue
    loader = PyPDFLoader(p)
    docs += loader.load()

# ─── 2) Split into chunks ───────────────────────────────────────
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

# ─── 3) Embedding using HuggingFace ─────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ─── 4) Store in Chroma ─────────────────────────────────────────
persist_dir = "chroma_db"
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_dir
)
vectordb.persist()

print(f"✅ Chroma DB built & saved to `{persist_dir}/` with {len(texts)} chunks.")
