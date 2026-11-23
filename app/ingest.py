import asyncio

from app.chunking import chunk_words
from app.db import connect
from app.embeddings import embed


def _save(title: str, source: str | None, chunks: list[str], vectors) -> int:
    with connect() as conn:                       # one transaction
        # Adding the same title again replaces the old version (chunks go too: ON DELETE CASCADE)
        conn.execute("DELETE FROM documents WHERE title = %s", (title,))
        row = conn.execute(
            "INSERT INTO documents (title, source) VALUES (%s, %s) RETURNING id",
            (title, source),
        ).fetchone()
        document_id = row[0]
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, content, embedding) "
                "VALUES (%s, %s, %s, %s)",
                [
                    (document_id, index, text, vector)
                    for index, (text, vector) in enumerate(zip(chunks, vectors))
                ],
            )
        return document_id


async def ingest_document(title: str, text: str, source: str | None = None) -> dict:
    chunks = chunk_words(text)
    if not chunks:
        raise ValueError("The document has no text")

    vectors = await embed(chunks, "search_document")          # async: talks to Ollama
    document_id = await asyncio.to_thread(_save, title, source, chunks, vectors)   # blocking DB work
    return {"document_id": document_id, "title": title, "chunks": len(chunks)}