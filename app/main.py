import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field

from app.extract import ExtractionError, extract_job
from app.schemas import ExtractRequest

from app.ingest import ingest_document
from app.rag import answer_question
from app.schemas import DocumentRequest, ExtractRequest, QuestionRequest


from app import llm

app = FastAPI(title="CareerPilot", description="A career guidance assistant", version="0.1.0")

class AskRequest(BaseModel):
    question: str = Field(min_length=10, max_length=2000, example="What are the best career options for a software engineer?")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=dict)
async def ask(req : AskRequest):
    """
    Ask a question to the LLM model and return the response.

    Args:
        request (Request): The FastAPI request object.
        question (str): The question to ask.
    """
    response = await llm.ask(req.question)
    return response


@app.post("/ask_stream")
async def ask_stream(req: AskRequest):
    async def event_stream():
        async for event in llm.ask_stream(req.question):
              yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(),media_type="text/event-stream")

@app.post("/extract")
async def extract(req: ExtractRequest):
    try:
        return await extract_job(req.text)
    except ExtractionError as error:
        raise HTTPException(status_code=502, detail=str(error))


@app.post("/documents")
async def add_document(req: DocumentRequest):
    return await ingest_document(req.title, req.text, req.source)


@app.post("/rag/ask")
async def rag_ask(req: QuestionRequest):
    return await answer_question(req.question)