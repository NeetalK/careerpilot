import time
from ollama import AsyncClient
from app.config import OLLAMA_MODEL

client = AsyncClient()

SYSTEM_PROMPT = "You are a helpful assistant. Answer in simple English."
OPTIONS = {"temperature": 0.2}

def build_messages(question: str) -> list[dict]:
    """
    Build the messages for the LLM model.

    Args:
        question (str): The question to ask.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    

async def ask(question: str) -> str:
    """
    Ask a question to the LLM model and return the response.

    Args:
        question (str): The question to ask.

    """
    start = time.perf_counter()
    response = await client.chat(
            model=OLLAMA_MODEL,
            messages=build_messages(question),
            options=OPTIONS,
    )
    return {
        "answer": response.message.content,
        "input_tokens": response.prompt_eval_count,
        "output_tokens": response.eval_count,
        "total_seconds": round(time.perf_counter() - start, 2),
    }



async def ask_stream(question: str) -> str:
    """
    Ask a question to the LLM model and return the response as a stream.

    Args:
        question (str): The question to ask.
    """
    start = time.perf_counter()
    first_token_seconds = None

    stream = await client.chat(
        model=OLLAMA_MODEL,
        messages=build_messages(question),
        options=OPTIONS,
        stream=True,
    )
    
    async for chunk in stream:
        piece = chunk.message.content
        if piece:
            if first_token_seconds is None:
                first_token_seconds = time.perf_counter() - start   
            yield {"type": "token", "text": piece}

            if chunk.done:
                yield {
                    "type": "done",
                    "input_tokens": response.prompt_eval_count,
                    "output_tokens": response.eval_count,
                    "first_token_seconds": round(first_token_seconds or 0, 2),
                    "total_seconds": round(time.perf_counter() - start, 2),
                }


if __name__ == "__main__":
    print(ask(input("Ask a question: ")))