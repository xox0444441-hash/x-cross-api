import os
import time
import httpx
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="X-CROSS API",
    version="1.2.0"
)

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

MODEL_ID = os.getenv(
    "X_CROSS_MODEL_ID",
    "gemini-3.1-flash-lite"
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/"
    + MODEL_ID
    + ":generateContent"
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
        "version": "1.2.0",
        "provider": "Google Gemini",
        "model": MODEL_ID
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():

    missing = []

    if not GOOGLE_API_KEY:
        missing.append("GOOGLE_API_KEY")

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
        "provider": "Google Gemini",
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

    # Check Google API Key
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Configure GOOGLE_API_KEY"
        )

    if data.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is not supported yet"
        )

    contents = []
    system_parts = []

    for message in data.messages:

        if not isinstance(message.content, str):
            raise HTTPException(
                status_code=422,
                detail="Message content must be a string"
            )

        if message.role == "system":

            system_parts.append({
                "text": message.content
            })

        elif message.role == "user":

            contents.append({
                "role": "user",
                "parts": [
                    {"text": message.content}
                ]
            })

        elif message.role == "assistant":

            contents.append({
                "role": "model",
                "parts": [
                    {"text": message.content}
                ]
            })

        else:

            raise HTTPException(
                status_code=422,
                detail="Unsupported message role"
            )

    if not contents:
        raise HTTPException(
            status_code=422,
            detail="A user message is required"
        )

    # Gemini Request Body
    body = {
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": data.max_tokens,
            "temperature": data.temperature
        }
    }

    if system_parts:
        body["systemInstruction"] = {
            "parts": system_parts
        }

    try:

        async with httpx.AsyncClient(
            timeout=120
        ) as client:

            response = await client.post(
                GEMINI_URL,
                json=body,
                headers={
                    "x-goog-api-key": GOOGLE_API_KEY,
                    "Content-Type": "application/json"
                }
            )

    except httpx.TimeoutException:

        raise HTTPException(
            status_code=504,
            detail="Google Gemini request timed out"
        )

    except httpx.RequestError:

        raise HTTPException(
            status_code=502,
            detail="Cannot reach Google Gemini"
        )

    if response.is_error:

        try:
            error_data = response.json()
            detail = error_data.get(
                "error", {}
            ).get(
                "message",
                response.text
            )
        except ValueError:
            detail = response.text

        raise HTTPException(
            status_code=response.status_code,
            detail=str(detail)[:2000]
        )

    try:

        result = response.json()

    except ValueError:

        raise HTTPException(
            status_code=502,
            detail="Invalid Gemini response"
        )

    candidates = result.get("candidates", [])

    if not candidates:

        reason = result.get(
            "promptFeedback", {}
        ).get(
            "blockReason",
            "No response returned"
        )

        raise HTTPException(
            status_code=502,
            detail=f"Gemini returned no answer: {reason}"
        )

    candidate = candidates[0]

    parts = candidate.get(
        "content", {}
    ).get(
        "parts", []
    )

    answer = "".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    )

    if not answer:

        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty response"
        )

    usage = result.get("usageMetadata", {})

    finish_reason = candidate.get(
        "finishReason", "STOP"
    )

    finish_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter"
    }

    # OpenAI Compatible Response
    return {
        "id": f"x-cross-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer
                },
                "finish_reason": finish_map.get(
                    finish_reason,
                    finish_reason.lower()
                )
            }
        ],
        "usage": {
            "prompt_tokens": usage.get(
                "promptTokenCount", 0
            ),
            "completion_tokens": usage.get(
                "candidatesTokenCount", 0
            ),
            "total_tokens": usage.get(
                "totalTokenCount", 0
            )
        }
    }
import
