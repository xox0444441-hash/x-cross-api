import os
import httpx
from typing import Any
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="X-CROSS API", version="1.0.0")

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_ID = os.getenv("X_CROSS_MODEL_ID", "")
HF_URL = os.getenv(
    "HF_API_BASE",
    "https://router.huggingface.co/v1/chat/completions"
)
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
    return {"name": "X-CROSS API", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ready")
async def ready():
    if not HF_TOKEN or not MODEL_ID:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready",
                     "missing": [
                         k for k, v in {
                             "HF_TOKEN": HF_TOKEN,
                             "X_CROSS_MODEL_ID": MODEL_ID
                         }.items() if not v
                     ]}
        )
    return {"status": "ready", "model": MODEL_ID}

@app.post("/v1/chat/completions")
async def chat(
    data: ChatRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    if X_API_KEY and x_api_key != X_API_KEY:
        raise HTTPException(401, "Invalid API key")

    if not HF_TOKEN or not MODEL_ID:
        raise HTTPException(
            503, "Configure HF_TOKEN and X_CROSS_MODEL_ID"
        )

    if data.stream:
        raise HTTPException(400, "Streaming is not supported yet")

    body = {
        "model": MODEL_ID,
        "messages": [m.model_dump() for m in data.messages],
        "max_tokens": data.max_tokens,
        "temperature": data.temperature,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                HF_URL,
                json=body,
                headers={"Authorization": f"Bearer {HF_TOKEN}"}
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "Model provider timed out")
    except httpx.RequestError:
        raise HTTPException(502, "Cannot reach model provider")

    if response.is_error:
        raise HTTPException(response.status_code, response.text[:2000])

    try:
        return response.json()
    except ValueError:
        raise HTTPException(502, "Invalid response from model provider")
