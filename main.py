import base64
import json
import logging
import os
import random
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Optional

import cv2
import fitz  # PyMuPDF
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ocr-service")

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GOOGLE_INVOKE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
OPENROUTER_INVOKE_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "nvidia/nemotron-3.5-super-vl-120b-a12b")
OPENROUTER_MODELS = [
    model.strip()
    for model in os.environ.get(
        "OPENROUTER_MODELS",
        "google/gemini-3.8-flash,google/gemini-3-flash-preview",
    ).split(",")
    if model.strip()
]
GOOGLE_MODELS = [
    model.strip()
    for model in os.environ.get(
        "GOOGLE_MODELS",
        "gemini-3.6-flash",
    ).split(",")
    if model.strip()
]
MODEL_PROVIDER_ORDER = [
    provider.strip().casefold()
    for provider in os.environ.get("MODEL_PROVIDER_ORDER", "google,nvidia,openrouter").split(",")
    if provider.strip()
]
NVIDIA_REQUEST_TIMEOUT = int(os.environ.get("NVIDIA_REQUEST_TIMEOUT", "45"))
MODEL_MAX_RETRIES = int(os.environ.get("MODEL_MAX_RETRIES", "2"))
RETRY_BASE_SECONDS = float(os.environ.get("RETRY_BASE_SECONDS", "0.5"))
LOG_MODEL_RAW_OUTPUT = os.environ.get("LOG_MODEL_RAW_OUTPUT", "false").casefold() == "true"
PDF_RENDER_DPI = int(os.environ.get("PDF_RENDER_DPI", "200"))
MAX_IMAGE_WIDTH = int(os.environ.get("MAX_IMAGE_WIDTH", "1800"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "10"))
MAX_RENDERED_PIXELS = int(os.environ.get("MAX_RENDERED_PIXELS", "25000000"))
MAX_CONCURRENT_OCR = int(os.environ.get("MAX_CONCURRENT_OCR", "4"))
MAX_OCR_SECONDS = int(os.environ.get("MAX_OCR_SECONDS", "180"))
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
APP_VERSION = "2026-11-08-production-fallbacks"
ocr_slots = threading.BoundedSemaphore(MAX_CONCURRENT_OCR)


class ModelProvidersUnavailable(RuntimeError):
    pass

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info(
        "startup: version=%s nvidia_models=%s google_models=%s openrouter_models=%s timeout=%s "
        "pdf_render_dpi=%s max_image_width=%s max_upload_bytes=%s max_pdf_pages=%s "
        "max_concurrent_ocr=%s raw_output_logs=%s",
        APP_VERSION,
        get_nvidia_models(),
        GOOGLE_MODELS,
        OPENROUTER_MODELS,
        NVIDIA_REQUEST_TIMEOUT,
        PDF_RENDER_DPI,
        MAX_IMAGE_WIDTH,
        MAX_UPLOAD_BYTES,
        MAX_PDF_PAGES,
        MAX_CONCURRENT_OCR,
        LOG_MODEL_RAW_OUTPUT,
    )
    yield


app = FastAPI(title="Radheshyam Prescription OCR Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials="*" not in CORS_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


EMPTY_RESPONSE = {
    "success": True,
    "prescriptionNumber": None,
    "patient": {
        "name": None,
        "address": None,
        "mobile": None,
        "gender": None,
        "age": None,
    },
    "medicines": [],
}


EXTRACTION_PROMPT = """
Extract only these details from this prescription image:

Name
Prescription
Address
Mobile
Gender
Age
Medicines with Qty

Return only valid JSON in this exact shape:
{
  "Name": null,
  "Prescription": null,
  "Address": null,
  "Mobile": null,
  "Gender": null,
  "Age": null,
  "Medicines": [
    {
      "Name": "",
      "Qty": null
    }
  ]
}

Rules:
- Use only text visible in the image.
- Do not guess.
- Do not add sample data.
- Do not infer missing values.
- Prescription means the prescription serial/number only. Prefer nearby labels like Sr.No, Sr No, Serial No, Prescription No, Rx No, Bill No, Slip No, or Order No.
- Do not use page number, card number, hospital, pharmacy, company, title, location, relation, category, diagnosis, or form text as Prescription.
- Name means patient name only. Prefer labels like Name or Patient Name.
- Address means patient address only. Prefer labels like Patient Address or Address.
- Mobile means the patient's phone/mobile number only. Accept it only when it is clearly associated with the patient or a patient field such as Mobile, Phone, Contact, Patient Mobile, or Patient Phone.
- Never use a phone number belonging to a beneficiary, attendant, relative, guardian, doctor, hospital, pharmacy, signature block, or Beneficiary/Attendant Signature section.
- If the only visible phone number is in or near a Beneficiary/Attendant Signature section, return Mobile as null.
- Gender means Sex or Gender.
- Age means Age.
- Medicines must come only from medicine table rows or prescription medicine lines. Prefer columns named Medicine Name, Medi.Name, Drug, Item, or Medicine.
- Qty must come only from columns or labels like Qty, Quantity, Qty-Prescribed, Qty-Presc., Day(s), or explicit medicine quantity text.
- Never use category, diagnosis, card, location, relation, doctor, chemist, pharmacy, form, route/form-only text, or hospital title as a medicine.
- If a field is missing or unclear, return null.
- If no medicines are visible, return an empty Medicines array.
- Qty must be a number when visible, otherwise null.
- Return JSON only. No markdown. No explanation.
""".strip()

def preprocess_image(image_bytes: bytes, request_id: str, page_label: str) -> bytes:
    started_at = time.perf_counter()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height * width > MAX_RENDERED_PIXELS:
        raise HTTPException(status_code=413, detail="Image dimensions are too large")
    logger.info(
        "[%s] preprocessing %s: original_size=%sx%s input_bytes=%s",
        request_id,
        page_label,
        width,
        height,
        len(image_bytes),
    )

    if width > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        logger.info("[%s] preprocessing %s: downscaled_with_scale=%.3f", request_id, page_label, scale)
    elif width < 1200:
        scale = 1200 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        logger.info("[%s] preprocessing %s: upscaled_with_scale=%.3f", request_id, page_label, scale)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    ok, encoded = cv2.imencode(".png", sharpened)
    if not ok:
        raise ValueError("Could not encode image")
    output = encoded.tobytes()
    logger.info(
        "[%s] preprocessing %s complete: output_bytes=%s elapsed_ms=%.1f",
        request_id,
        page_label,
        len(output),
        (time.perf_counter() - started_at) * 1000,
    )
    return output


def file_to_page_inputs(file_bytes: bytes, content_type: Optional[str], request_id: str) -> list[dict[str, Any]]:
    started_at = time.perf_counter()
    is_pdf = content_type == "application/pdf" or file_bytes.startswith(b"%PDF")
    logger.info(
        "[%s] converting upload: content_type=%s bytes=%s is_pdf=%s",
        request_id,
        content_type,
        len(file_bytes),
        is_pdf,
    )

    if not is_pdf:
        pages = [{"image": preprocess_image(file_bytes, request_id, "image"), "text": None}]
        logger.info(
            "[%s] image conversion complete: pages=%s elapsed_ms=%.1f",
            request_id,
            len(pages),
            (time.perf_counter() - started_at) * 1000,
        )
        return pages

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc

    if len(doc) == 0:
        doc.close()
        raise HTTPException(status_code=400, detail="PDF contains no pages")
    if len(doc) > MAX_PDF_PAGES:
        doc.close()
        raise HTTPException(status_code=413, detail=f"PDF exceeds the {MAX_PDF_PAGES}-page limit")

    logger.info("[%s] pdf opened: pages=%s", request_id, len(doc))
    pages = []
    try:
        for page_num in range(len(doc)):
            page_started_at = time.perf_counter()
            page = doc.load_page(page_num)
            scale = PDF_RENDER_DPI / 72
            estimated_pixels = int(page.rect.width * scale) * int(page.rect.height * scale)
            if estimated_pixels > MAX_RENDERED_PIXELS:
                raise HTTPException(status_code=413, detail="PDF page dimensions are too large")

            page_text = re.sub(r"\s+", " ", page.get_text("text")).strip() or None
            logger.info(
                "[%s] extracted pdf text page %s: chars=%s",
                request_id,
                page_num + 1,
                len(page_text or ""),
            )
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI)
            logger.info(
                "[%s] rendered pdf page %s: dpi=%s pixmap=%sx%s elapsed_ms=%.1f",
                request_id,
                page_num + 1,
                PDF_RENDER_DPI,
                pix.width,
                pix.height,
                (time.perf_counter() - page_started_at) * 1000,
            )
            pages.append(
                {
                    "image": preprocess_image(pix.tobytes("png"), request_id, f"page_{page_num + 1}"),
                    "text": page_text,
                }
            )
    finally:
        doc.close()
    logger.info(
        "[%s] pdf conversion complete: pages=%s elapsed_ms=%.1f",
        request_id,
        len(pages),
        (time.perf_counter() - started_at) * 1000,
    )
    return pages


def extract_json_object(text: Optional[str]) -> dict[str, Any]:
    if not isinstance(text, str) or not text:
        return {}
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def get_nvidia_models() -> list[str]:
    return [NVIDIA_MODEL] if NVIDIA_MODEL else []


def build_model_payload(
    model: str,
    image_base64: str,
    page_text: Optional[str],
    use_json_mode: bool,
    provider: str,
) -> dict[str, Any]:
    prompt = EXTRACTION_PROMPT
    if page_text:
        prompt = (
            f"{EXTRACTION_PROMPT}\n\n"
            "Additional OCR/PDF text extracted directly from this same page. "
            "Use it only as supporting evidence for text visible in the image:\n"
            f"{page_text}"
        )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }

    if provider == "google":
        payload.pop("temperature", None)
        payload.pop("top_p", None)
        payload["max_tokens"] = 4096

    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    if provider == "nvidia" and model.startswith("nvidia/nemotron-3.5-"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    if provider == "openrouter":
        payload["provider"] = {"allow_fallbacks": True}
        payload["reasoning"] = (
            {"effort": "low"}
            if model == "google/gemini-3.8-flash"
            else {"enabled": False}
        )

    return payload


def call_vision_models(
    image_bytes: bytes,
    page_text: Optional[str],
    request_id: str,
    page_number: int,
    deadline: Optional[float] = None,
) -> dict[str, Any]:
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    available_providers = {}
    nvidia_key = os.environ.get("NVIDIA_API_KEY")
    google_key = os.environ.get("GEMINI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if nvidia_key:
        available_providers["nvidia"] = ("nvidia", NVIDIA_INVOKE_URL, nvidia_key, get_nvidia_models())
    if google_key and GOOGLE_MODELS:
        available_providers["google"] = ("google", GOOGLE_INVOKE_URL, google_key, GOOGLE_MODELS)
    if openrouter_key and OPENROUTER_MODELS:
        available_providers["openrouter"] = (
            "openrouter",
            OPENROUTER_INVOKE_URL,
            openrouter_key,
            OPENROUTER_MODELS,
        )
    providers = [available_providers[name] for name in MODEL_PROVIDER_ORDER if name in available_providers]
    if not providers:
        raise ModelProvidersUnavailable("No model provider API key is configured")

    logger.info(
        "[%s] page %s model call prepared: image_bytes=%s providers=%s timeout_seconds=%s",
        request_id,
        page_number,
        len(image_bytes),
        [provider[0] for provider in providers],
        NVIDIA_REQUEST_TIMEOUT,
    )
    logger.info(
        "[%s] page %s text context: chars=%s",
        request_id,
        page_number,
        len(page_text or ""),
    )
    errors = []
    for provider, url, api_key, models in providers:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if provider == "google":
            headers["x-goog-api-client"] = f"radheshyam-ocr/{APP_VERSION}"
        for model in models:
            for use_json_mode in (True, False):
                payload = build_model_payload(model, image_base64, page_text, use_json_mode, provider)
                for retry in range(MODEL_MAX_RETRIES + 1):
                    if deadline is not None and time.perf_counter() >= deadline:
                        raise ModelProvidersUnavailable("OCR request deadline exceeded")
                    started_at = time.perf_counter()
                    read_timeout = NVIDIA_REQUEST_TIMEOUT
                    if deadline is not None:
                        read_timeout = max(1, min(read_timeout, deadline - started_at))
                    try:
                        response = requests.post(
                            url,
                            headers=headers,
                            json=payload,
                            timeout=(5, read_timeout),
                        )
                        elapsed_ms = (time.perf_counter() - started_at) * 1000
                        logger.info(
                            "[%s] page %s model response: provider=%s model=%s json_mode=%s "
                            "status=%s retry=%s elapsed_ms=%.1f",
                            request_id,
                            page_number,
                            provider,
                            model,
                            use_json_mode,
                            response.status_code,
                            retry,
                            elapsed_ms,
                        )
                        if response.status_code in {408, 429, 500, 502, 503, 504}:
                            raise requests.HTTPError(response=response)
                        response.raise_for_status()

                        data = response.json()
                        choices = data.get("choices") or []
                        content = choices[0].get("message", {}).get("content") if choices else None
                        parsed = extract_json_object(content)
                        logger.info(
                            "[%s] page %s model usage: provider=%s model=%s usage=%s",
                            request_id,
                            page_number,
                            provider,
                            data.get("model", model),
                            json.dumps(data.get("usage"), ensure_ascii=False) if data.get("usage") else None,
                        )
                        if LOG_MODEL_RAW_OUTPUT:
                            logger.info(
                                "[%s] page %s model raw output: provider=%s model=%s content=%s",
                                request_id,
                                page_number,
                                provider,
                                model,
                                content,
                            )
                        if parsed:
                            return parsed

                        errors.append(f"{provider}/{model}: invalid JSON")
                        logger.warning(
                            "[%s] page %s unusable model output: provider=%s model=%s",
                            request_id,
                            page_number,
                            provider,
                            model,
                        )
                        break
                    except (requests.Timeout, requests.ConnectionError) as exc:
                        errors.append(f"{provider}/{model}: {type(exc).__name__}")
                        if retry >= MODEL_MAX_RETRIES:
                            break
                    except requests.HTTPError as exc:
                        status = exc.response.status_code if exc.response is not None else None
                        errors.append(f"{provider}/{model}: HTTP {status}")
                        if status not in {408, 429, 500, 502, 503, 504} or retry >= MODEL_MAX_RETRIES:
                            break
                        retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                        if retry_after and retry_after.isdigit():
                            time.sleep(min(float(retry_after), 10))
                            continue
                    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                        errors.append(f"{provider}/{model}: {type(exc).__name__}")
                        break

                    delay = RETRY_BASE_SECONDS * (2**retry) + random.uniform(0, RETRY_BASE_SECONDS)
                    time.sleep(min(delay, 10))

    logger.error("[%s] page %s all model providers failed: errors=%s", request_id, page_number, errors)
    raise ModelProvidersUnavailable("All configured vision models failed")


def clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None

    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def clean_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    medicines = []
    raw_medicines = result.get("Medicines") or result.get("medicines") or []

    if isinstance(raw_medicines, list):
        for medicine in raw_medicines:
            if not isinstance(medicine, dict):
                continue
            name = clean_string(medicine.get("Name") or medicine.get("name"))
            if not name:
                continue
            medicines.append(
                {
                    "Name": name,
                    "Qty": clean_int(medicine.get("Qty") or medicine.get("qty") or medicine.get("quantity")),
                }
            )

    return {
        "success": True,
        "Name": clean_string(result.get("Name") or result.get("name")),
        "Prescription": clean_string(result.get("Prescription") or result.get("prescription")),
        "Address": clean_string(result.get("Address") or result.get("address")),
        "Mobile": clean_string(result.get("Mobile") or result.get("mobile")),
        "Gender": clean_string(result.get("Gender") or result.get("gender")),
        "Age": clean_int(result.get("Age") or result.get("age")),
        "Medicines": medicines,
    }


def merge_page_results(page_results: list[dict[str, Any]], request_id: str) -> dict[str, Any]:
    logger.info("[%s] merging page results: pages=%s", request_id, len(page_results))
    merged = {
        "success": True,
        "Name": None,
        "Prescription": None,
        "Address": None,
        "Mobile": None,
        "Gender": None,
        "Age": None,
        "Medicines": [],
    }
    seen_medicines = set()

    for result in page_results:
        normalized = normalize_result(result)
        for field in ("Name", "Prescription", "Address", "Mobile", "Gender", "Age"):
            if merged[field] is None and normalized[field] is not None:
                merged[field] = normalized[field]

        for medicine in normalized["Medicines"]:
            key = (medicine["Name"].casefold(), medicine["Qty"])
            if key not in seen_medicines:
                merged["Medicines"].append(medicine)
                seen_medicines.add(key)

    logger.info("[%s] merged result: medicines=%s", request_id, len(merged["Medicines"]))
    return merged


def build_frontend_response(result: dict[str, Any], request_id: str) -> dict[str, Any]:
    medicines = [
        {
            "id": None,
            "name": medicine["Name"],
            "price": None,
            "stock": None,
            "quantity": medicine["Qty"],
            "confidence": 1.0,
        }
        for medicine in result["Medicines"]
    ]

    data = {
        "prescriptionNumber": result["Prescription"],
        "patient": {
            "name": result["Name"],
            "address": result["Address"],
            "mobile": result["Mobile"],
            "gender": result["Gender"],
            "age": result["Age"],
        },
        "medicines": medicines,
    }

    response = {
        "success": True,
        **data,
    }
    logger.info("[%s] frontend response built: medicines=%s", request_id, len(medicines))
    return response


@app.post("/ocr")
def run_ocr(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    deadline = started_at + MAX_OCR_SECONDS
    logger.info(
        "[%s] /ocr request start: content_type=%s",
        request_id,
        file.content_type,
    )

    if not ocr_slots.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail={"error": "OCR service is busy", "requestId": request_id},
            headers={"Retry-After": "2"},
        )
    try:
        file_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
        if not file_bytes:
            raise HTTPException(status_code=400, detail={"error": "Uploaded file is empty", "requestId": request_id})
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail={"error": f"Upload exceeds the {MAX_UPLOAD_BYTES}-byte limit", "requestId": request_id},
            )

        is_pdf = file_bytes.startswith(b"%PDF")
        is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
        is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_webp = len(file_bytes) >= 12 and file_bytes[:4] == b"RIFF" and file_bytes[8:12] == b"WEBP"
        if not any((is_pdf, is_jpeg, is_png, is_webp)):
            raise HTTPException(
                status_code=415,
                detail={"error": "Only PDF, JPG, PNG, and WEBP files are supported", "requestId": request_id},
            )

        logger.info("[%s] upload validated: bytes=%s", request_id, len(file_bytes))
        page_inputs = file_to_page_inputs(file_bytes, file.content_type, request_id)
        if not page_inputs:
            raise HTTPException(
                status_code=422,
                detail={"error": "Unable to read the prescription", "requestId": request_id},
            )

        page_results = []
        for index, page_input in enumerate(page_inputs, start=1):
            logger.info("[%s] starting OCR for page %s/%s", request_id, index, len(page_inputs))
            page_results.append(
                call_vision_models(
                    page_input["image"],
                    page_input.get("text"),
                    request_id,
                    index,
                    deadline,
                )
            )

        result = merge_page_results(page_results, request_id)

        has_any_data = any(result[field] is not None for field in ("Name", "Prescription", "Address", "Mobile", "Gender", "Age"))
        if not has_any_data and not result["Medicines"]:
            logger.warning("[%s] no extractable data after merge", request_id)
            raise HTTPException(
                status_code=422,
                detail={"error": "No prescription data could be extracted", "requestId": request_id},
            )

        response = build_frontend_response(result, request_id)
        logger.info("[%s] /ocr request complete: elapsed_ms=%.1f", request_id, (time.perf_counter() - started_at) * 1000)
        return response
    except HTTPException as exc:
        logger.warning("[%s] /ocr request rejected: status=%s", request_id, exc.status_code)
        raise
    except ModelProvidersUnavailable:
        logger.exception("[%s] all model providers unavailable", request_id)
        raise HTTPException(
            status_code=503,
            detail={"error": "OCR providers are temporarily unavailable", "requestId": request_id},
            headers={"Retry-After": "5"},
        )
    except Exception:
        logger.exception("[%s] /ocr request failed", request_id)
        raise HTTPException(
            status_code=500,
            detail={"error": "OCR processing failed", "requestId": request_id},
        )
    finally:
        ocr_slots.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    configured = bool(
        os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not configured:
        raise HTTPException(status_code=503, detail="No OCR model provider is configured")
    return {"status": "ready"}


@app.get("/")
def root():
    return {"status": "ok", "version": APP_VERSION}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    workers = 1 if os.name == "nt" else int(os.environ.get("WEB_CONCURRENCY", "1"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers)
