
import os
import httpx
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="X-CROSS API",
    version="1.1.0"
)

# DeepInfra Configuration
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "")
MODEL_ID = os.getenv(
    "X_CROSS_MODEL_ID",
    "Qwen/Qwen3.5-9B"
)

DEEPINFRA_URL = (
    "https://api.deepinfra.com/v1/openai/chat/completions"
)

# Optional X-CROSS API security key
X_API_KEY = os.getenv("X_API_KEY", "")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    stream: bool = False


@app.get("/")
async def home():
    return {
        "name": "X-CROSS API",
        "status": "online",
        "version": "1.1.0"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():

    missing = []

    if not DEEPINFRA_API_KEY:
        missing.append("DEEPINFRA_API_KEY")

    if not MODEL_ID:
        missing.append("X_CROSS_MODEL_ID")

    if missing:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "missing": missing
            }
        )

    return {
        "status": "ready",
        "provider": "DeepInfra",
        "model": MODEL_ID
    }


@app.post("/v1/chat/completions")
async def chat(
    data: ChatRequest,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key"
    )
):

    # Check X-CROSS API Key
    if X_API_KEY and x_api_key != X_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    # Check DeepInfra API Key
    if not DEEPINFRA_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Configure DEEPINFRA_API_KEY"
        )

    if data.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported yet"
        )

    body = {
        "model": MODEL_ID,
        "messages": [
            message.model_dump()
            for message in data.messages
        ],
        "max_tokens": data.max_tokens,
        "temperature": data.temperature,
        "stream": False
    }

    try:

        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            response = await client.post(
                DEEPINFRA_URL,
                json=body,
                headers={
                    "Authorization": (
                        f"Bearer {DEEPINFRA_API_KEY}"
                    ),
                    "Content-Type": "application/json"
                }
            )

    except httpx.TimeoutException:

        raise HTTPException(
            status_code=504,
            detail="DeepInfra request timed out"
        )

    except httpx.RequestError:

        raise HTTPException(
            status_code=502,
            detail="Cannot reach DeepInfra"
        )

    if response.is_error:

        raise HTTPException(
            status_code=response.status_code,
            detail=response.text[:2000]
        )

    try:

        return response.json()

    except ValueError:

        raise HTTPException(
            status_code=502,
            detail="Invalid response from DeepInfra"
        )
