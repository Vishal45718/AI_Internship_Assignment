from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.router import router
from src.config import reload_settings
from src.llm.client import validate_gemini_startup_or_raise, validate_ollama_startup_or_raise

app = FastAPI(title="RAG Backend System", version="1.0.0")

# Setup CORS to allow the Next.js frontend to communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.on_event("startup")
def validate_provider_on_startup() -> None:
    settings = reload_settings()
    if settings.llm_provider == "gemini":
        resolved_model, available_models = validate_gemini_startup_or_raise(
            settings.gemini_api_key,
            settings.gemini_model,
        )
        print(f"Active provider={settings.llm_provider} model={resolved_model}")
        if resolved_model != settings.gemini_model:
            print(f"Configured GEMINI_MODEL={settings.gemini_model} was unavailable; using {resolved_model}")
        print(f"Available Gemini models ({len(available_models)}): {', '.join(available_models[:20])}")
        return
    if settings.llm_provider == "ollama":
        resolved_model, available_models = validate_ollama_startup_or_raise(
            settings.ollama_base_url,
            settings.ollama_model,
        )
        print(f"Active provider={settings.llm_provider} model={resolved_model}")
        print(f"Available Ollama models ({len(available_models)}): {', '.join(available_models[:20])}")
        return
    raise RuntimeError(
        f"Invalid LLM_PROVIDER='{settings.llm_provider}'. Supported values: gemini, ollama."
    )

@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAG Backend is running"}
