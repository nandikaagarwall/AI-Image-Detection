"""Text-to-image backends used by the Streamlit app."""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

from PIL import Image

GEMINI_IMAGE_MODELS = [
    "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview",
    "gemini-2.0-flash-preview-image-generation",
]


class GenerationError(RuntimeError):
    pass


def load_api_key(env_file: str | Path | None = None) -> str:
    """Gemini key from the environment or a .env file next to this script.
    Accepts GEMINI_API_KEY / GOOGLE_API_KEY in any letter case (e.g. Gemini_Api_Key)."""
    wanted = {"GEMINI_API_KEY", "GOOGLE_API_KEY"}
    for k, v in os.environ.items():
        if k.upper() in wanted and v.strip():
            return v.strip()
    path = Path(env_file) if env_file else Path(__file__).parent / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip().removeprefix("export ").strip()
            if k.upper() in wanted:
                return v.strip().strip('"').strip("'")
    return ""


def generate_gemini(api_key: str, prompt: str, model: str) -> Image.Image:
    """Generate an image with a Gemini native image model (google-genai SDK)."""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except errors.APIError as e:
        hint = ""
        if getattr(e, "code", None) == 429:
            hint = (
                "\n\nHTTP 429 = quota exhausted. If the message says 'limit: 0', this image model "
                "has no free-tier quota for your key/project - use the 'Pollinations (free, no key)' "
                "backend or upload an image instead."
            )
        elif getattr(e, "code", None) == 404:
            hint = "\n\nModel not found for your key - click 'List image models for my key'."
        raise GenerationError(f"Gemini error {getattr(e, 'code', '')}: {e.message}{hint}") from e

    texts = []
    for cand in resp.candidates or []:
        for part in (cand.content.parts if cand.content else None) or []:
            if part.inline_data is not None and part.inline_data.data:
                return Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            if part.text:
                texts.append(part.text)
    raise GenerationError(
        "Gemini returned no image." + (f" Model said: {' '.join(texts)[:500]}" if texts else "")
    )


def list_gemini_image_models(api_key: str) -> list[str]:
    from google import genai

    client = genai.Client(api_key=api_key)
    names = []
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        if "image" in name or "imagen" in name:
            names.append(name)
    return sorted(names)


def generate_pollinations(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42) -> Image.Image:
    """Free keyless backend (pollinations.ai) - handy when Gemini's free tier has no image quota."""
    import requests

    url = (
        f"https://image.pollinations.ai/prompt/{quote(prompt, safe='')}"
        f"?width={width}&height={height}&seed={seed}&nologo=true"
    )
    r = requests.get(url, timeout=120)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
        raise GenerationError(f"Pollinations error {r.status_code}: {r.text[:300]}")
    return Image.open(BytesIO(r.content)).convert("RGB")
