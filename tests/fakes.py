from types import SimpleNamespace


class FakeClient:
    """Pretends to be the Ollama client. Returns the replies we give it, in order."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, **kwargs):
        # Save a copy of what we were asked, so tests can check it later
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        raw = self.replies.pop(0)
        return SimpleNamespace(
            message=SimpleNamespace(content=raw),
            prompt_eval_count=10,
            eval_count=5,
        )