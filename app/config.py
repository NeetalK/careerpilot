import os
from dotenv import load_dotenv

load_dotenv()   

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# RAG settings
CHUNK_WORDS = 200        # words per chunk
CHUNK_OVERLAP = 40       # words repeated between neighbouring chunks
TOP_K = 4                # how many nearest chunks to fetch
MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.55"))   # farther than this = not relevant