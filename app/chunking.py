from app.config import CHUNK_OVERLAP, CHUNK_WORDS


def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of `size` words. Neighbouring chunks share `overlap` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):   # this chunk already reaches the end
            break
    return chunks