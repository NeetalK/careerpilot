import numpy as np

from app.config import EMBED_MODEL
from app.llm import client


async def embed(texts: list[str], kind: str) -> list[np.ndarray]:
    """kind is 'search_document' for stored text and 'search_query' for questions."""
    prefixed = [f"{kind}: {text}" for text in texts]
    response = await client.embed(model=EMBED_MODEL, input=prefixed)
    return [np.array(vector) for vector in response.embeddings]