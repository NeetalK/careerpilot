import os

import numpy as np
import ollama
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()
DATABASE_URL = os.environ["DATABASE_URL"]
EMBED_MODEL = "nomic-embed-text"

SENTENCES = [
    "I have 7 years of experience building APIs with Python and FastAPI.",
    "The company offers 25 days of holiday and a learning budget.",
    "We are looking for a backend engineer who knows PostgreSQL and Docker.",
    "The office is in Riga and employees work three days on site.",
    "Experience with Kubernetes and cloud platforms is a plus.",
]


def embed(texts: list[str], kind: str) -> list[np.ndarray]:
    # nomic-embed-text expects a prefix that says what the text is
    prefixed = [f"{kind}: {text}" for text in texts]
    response = ollama.embed(model=EMBED_MODEL, input=prefixed)
    return [np.array(vector) for vector in response.embeddings]


def main():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        register_vector(conn)

        conn.execute("DROP TABLE IF EXISTS demo_sentences")
        conn.execute(
            "CREATE TABLE demo_sentences ("
            "id serial PRIMARY KEY, content text, embedding vector(768))"
        )

        vectors = embed(SENTENCES, "search_document")
        print("Dimension of one vector:", len(vectors[0]))

        for text, vector in zip(SENTENCES, vectors):
            conn.execute(
                "INSERT INTO demo_sentences (content, embedding) VALUES (%s, %s)",
                (text, vector),
            )
        print(f"Stored {len(SENTENCES)} sentences. Ask a question (empty line to stop).\n")

        while True:
            question = input("Question: ").strip()
            if not question:
                break
            query_vector = embed([question], "search_query")[0]
            rows = conn.execute(
                "SELECT content, embedding <=> %s AS distance "
                "FROM demo_sentences ORDER BY distance LIMIT 3",
                (query_vector,),
            ).fetchall()
            for content, distance in rows:
                print(f"  {distance:.3f}  {content}")
            print()


main()