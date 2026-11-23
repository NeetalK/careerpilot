import asyncio
import sys
from pathlib import Path

from app.ingest import ingest_document


async def main(folder: str):
    files = sorted(Path(folder).glob("*.txt")) + sorted(Path(folder).glob("*.md"))
    if not files:
        print("No .txt or .md files found in", folder)
        return
    for path in files:
        text = path.read_text(encoding="utf-8")
        result = await ingest_document(path.stem, text, source=str(path))
        print(f"{path.name}: {result['chunks']} chunks")


asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "data"))