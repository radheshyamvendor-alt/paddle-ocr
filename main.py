import base64
import json
import logging
import os
import re
import time
import uuid
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
NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")
NVIDIA_FALLBACK_MODELS = [
    model.strip()
    for model in os.environ.get("NVIDIA_FALLBACK_MODELS", "meta/llama-3.2-90b-vision-instruct").split(",")
    if model.strip()
]
NVIDIA_REQUEST_TIMEOUT = int(os.environ.get("NVIDIA_REQUEST_TIMEOUT", "45"))
LOG_MODEL_RAW_OUTPUT = os.environ.get("LOG_MODEL_RAW_OUTPUT", "true").casefold() == "true"

app = FastAPI(title="Radheshyam NVIDIA NIM Vision OCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
- Prescription means the prescription serial/number only. Prefer labels like Sr.No, Sr No, Serial No, Prescription No, Rx No, Bill No, Slip No, or Order No.
- For ONGC-style slips, Prescription is Sr.No, for example 1000020419. Never use MSB-PAGE NO, page number, 2.0, card number, hospital, pharmacy, company, title, or location text as Prescription.
- Name means patient name only. Prefer labels like Name or Patient Name.
- Address means patient address only. Prefer labels like Patient Address or Address.
- Mobile means a phone/mobile number visible anywhere on the prescription. Use a visible 10-digit phone number if present, even if it appears near signature or attendant details.
- Gender means Sex or Gender.
- Age means Age.
- Medicines must come only from medicine table rows or prescription medicine lines. Prefer columns named Medicine Name, Medi.Name, Drug, Item, or Medicine.
- Qty must come only from columns or labels like Qty, Quantity, Qty-Prescribed, Qty-Presc., Day(s), or explicit medicine quantity text.
- Never use Emp Catg, CAT EMP, Diagnosis, Card No, Location, Relation, Doctor Name, Chemist, Pharmacy, Form, TAB alone, CAP alone, or hospital title as a medicine.
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
    logger.info(
        "[%s] preprocessing %s: original_size=%sx%s input_bytes=%s",
        request_id,
        page_label,
        width,
        height,
        len(image_bytes),
    )

    if width < 1600:
        scale = 1600 / width
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        logger.info("[%s] preprocessing %s: resized_with_scale=%.3f", request_id, page_label, scale)

    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
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


def file_to_page_images(file_bytes: bytes, content_type: Optional[str], request_id: str) -> list[bytes]:
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
        images = [preprocess_image(file_bytes, request_id, "image")]
        logger.info(
            "[%s] image conversion complete: pages=%s elapsed_ms=%.1f",
            request_id,
            len(images),
            (time.perf_counter() - started_at) * 1000,
        )
        return images

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid PDF: {exc}") from exc

    logger.info("[%s] pdf opened: pages=%s", request_id, len(doc))
    images = []
    for page_num in range(len(doc)):
        page_started_at = time.perf_counter()
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        logger.info(
            "[%s] rendered pdf page %s: pixmap=%sx%s elapsed_ms=%.1f",
            request_id,
            page_num + 1,
            pix.width,
            pix.height,
            (time.perf_counter() - page_started_at) * 1000,
        )
        images.append(preprocess_image(pix.tobytes("png"), request_id, f"page_{page_num + 1}"))
    logger.info(
        "[%s] pdf conversion complete: pages=%s elapsed_ms=%.1f",
        request_id,
        len(images),
        (time.perf_counter() - started_at) * 1000,
    )
    return images


def extract_json_object(text: str) -> dict[str, Any]:
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
    models = []
    for model in [NVIDIA_MODEL, *NVIDIA_FALLBACK_MODELS]:
        if model and model not in models:
            models.append(model)
    return models


def build_nvidia_payload(model: str, image_base64: str, use_json_mode: bool) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
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

    if use_json_mode:
        payload["response_format"] = {"type": "json_object"}

    return payload


def call_nvidia_vision(image_bytes: bytes, request_id: str, page_number: int) -> dict[str, Any]:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is missing. Add it to .env")

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    logger.info(
        "[%s] page %s nvidia call prepared: image_bytes=%s image_base64_chars=%s models=%s timeout_seconds=%s",
        request_id,
        page_number,
        len(image_bytes),
        len(image_base64),
        get_nvidia_models(),
        NVIDIA_REQUEST_TIMEOUT,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    errors = []
    for model in get_nvidia_models():
        for use_json_mode in (True, False):
            payload = build_nvidia_payload(model, image_base64, use_json_mode)
            started_at = time.perf_counter()
            logger.info(
                "[%s] page %s nvidia attempt start: model=%s json_mode=%s",
                request_id,
                page_number,
                model,
                use_json_mode,
            )
            try:
                response = requests.post(
                    NVIDIA_INVOKE_URL,
                    headers=headers,
                    json=payload,
                    timeout=NVIDIA_REQUEST_TIMEOUT,
                )
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "[%s] page %s nvidia attempt response: model=%s json_mode=%s status=%s elapsed_ms=%.1f",
                    request_id,
                    page_number,
                    model,
                    use_json_mode,
                    response.status_code,
                    elapsed_ms,
                )
                response.raise_for_status()

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage")
                logger.info(
                    "[%s] page %s nvidia usage: model=%s usage=%s",
                    request_id,
                    page_number,
                    model,
                    json.dumps(usage, ensure_ascii=False) if usage else None,
                )
                if LOG_MODEL_RAW_OUTPUT:
                    logger.info(
                        "[%s] page %s nvidia raw output: model=%s content=%s",
                        request_id,
                        page_number,
                        model,
                        content,
                    )
                parsed = extract_json_object(content)
                if parsed:
                    logger.info(
                        "[%s] page %s nvidia parsed output: model=%s parsed=%s",
                        request_id,
                        page_number,
                        model,
                        json.dumps(parsed, ensure_ascii=False),
                    )
                    return parsed

                errors.append(f"{model}: empty JSON response")
                logger.warning(
                    "[%s] page %s nvidia attempt failed: model=%s reason=empty_json_response",
                    request_id,
                    page_number,
                    model,
                )
                break
            except requests.Timeout:
                errors.append(f"{model}: request timed out")
                logger.warning(
                    "[%s] page %s nvidia attempt timed out: model=%s json_mode=%s timeout_seconds=%s elapsed_ms=%.1f",
                    request_id,
                    page_number,
                    model,
                    use_json_mode,
                    NVIDIA_REQUEST_TIMEOUT,
                    (time.perf_counter() - started_at) * 1000,
                )
                break
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                response_text = exc.response.text if exc.response is not None else ""
                errors.append(f"{model}: HTTP {status_code}")
                logger.warning(
                    "[%s] page %s nvidia attempt http error: model=%s json_mode=%s status=%s body=%s",
                    request_id,
                    page_number,
                    model,
                    use_json_mode,
                    status_code,
                    response_text,
                )
                if status_code not in {400, 422} or not use_json_mode:
                    break
            except requests.RequestException as exc:
                errors.append(f"{model}: {exc}")
                logger.warning(
                    "[%s] page %s nvidia attempt request error: model=%s json_mode=%s error=%s",
                    request_id,
                    page_number,
                    model,
                    use_json_mode,
                    exc,
                )
                break

    logger.error("[%s] page %s all nvidia models failed: errors=%s", request_id, page_number, errors)
    raise RuntimeError("All NVIDIA vision models failed: " + "; ".join(errors))


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
    logger.info(
        "[%s] merging page results: page_results=%s",
        request_id,
        json.dumps(page_results, ensure_ascii=False),
    )
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
        logger.info("[%s] normalized page result: %s", request_id, json.dumps(normalized, ensure_ascii=False))
        for field in ("Name", "Prescription", "Address", "Mobile", "Gender", "Age"):
            if merged[field] is None and normalized[field] is not None:
                merged[field] = normalized[field]

        for medicine in normalized["Medicines"]:
            key = (medicine["Name"].casefold(), medicine["Qty"])
            if key not in seen_medicines:
                merged["Medicines"].append(medicine)
                seen_medicines.add(key)

    logger.info("[%s] merged result: %s", request_id, json.dumps(merged, ensure_ascii=False))
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
        "data": data,
    }
    logger.info("[%s] frontend response: %s", request_id, json.dumps(response, ensure_ascii=False))
    return response


@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:12]
    started_at = time.perf_counter()
    logger.info(
        "[%s] /ocr request start: filename=%s content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )
    file_bytes = await file.read()
    logger.info("[%s] upload read complete: bytes=%s", request_id, len(file_bytes))

    try:
        page_images = file_to_page_images(file_bytes, file.content_type, request_id)
        if not page_images:
            logger.warning("[%s] no page images generated", request_id)
            return {"success": False, "error": "Unable to extract text from prescription"}

        page_results = []
        for index, image in enumerate(page_images, start=1):
            logger.info("[%s] starting OCR for page %s/%s", request_id, index, len(page_images))
            page_results.append(call_nvidia_vision(image, request_id, index))

        result = merge_page_results(page_results, request_id)

        has_any_data = any(result[field] is not None for field in ("Name", "Prescription", "Address", "Mobile", "Gender", "Age"))
        if not has_any_data and not result["Medicines"]:
            logger.warning("[%s] no extractable data after merge", request_id)
            return {"success": False, "error": "Unable to extract text from prescription"}

        response = build_frontend_response(result, request_id)
        logger.info("[%s] /ocr request complete: elapsed_ms=%.1f", request_id, (time.perf_counter() - started_at) * 1000)
        return response
    except HTTPException:
        logger.exception("[%s] /ocr request failed with HTTPException", request_id)
        raise
    except requests.HTTPError as exc:
        logger.exception("[%s] /ocr request failed with NVIDIA HTTPError", request_id)
        return {"success": False, "error": f"NVIDIA OCR request failed: {exc}"}
    except Exception as exc:
        logger.exception("[%s] /ocr request failed", request_id)
        return {"success": False, "error": f"OCR extraction failed: {exc}"}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
