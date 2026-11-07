# Radheshyam Prescription OCR Microservice

Production-oriented FastAPI service that extracts prescription data with NVIDIA NIM and falls back to direct Google Gemini and OpenRouter vision models.

## Local Setup

Use Python 3.10 or 3.11 for this project.

Create a `.env` file from `.env.example` and set your NVIDIA API key:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`. Each provider requires its own key. One Google key covers all direct Gemini models, and one OpenRouter key covers every configured OpenRouter model:

```text
NVIDIA_API_KEY=your_real_key
NVIDIA_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
GEMINI_API_KEY=your_google_ai_studio_key
GOOGLE_MODELS=gemini-3.6-flash
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODELS=google/gemini-3.8-flash,google/gemini-3-flash-preview
MODEL_PROVIDER_ORDER=google,nvidia,openrouter
```

The default order is direct Google Gemini 3.6 Flash, NVIDIA Nemotron 3.5 Super VL, then the OpenRouter models in their configured order. `MODEL_PROVIDER_ORDER` changes provider priority, while each multi-model provider's model list controls its internal order. Temporary network, rate-limit, and server failures are retried with exponential backoff. OpenRouter also performs provider-level failover.

From `E:\OCR\paddle-ocr`:

```powershell
..\venv\Scripts\python.exe -m ensurepip --upgrade
..\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe main.py
```

Direct Windows execution uses one Uvicorn worker because Uvicorn multiprocess socket sharing is not reliable on Windows. Docker/Linux also defaults to one worker and can be overridden with `WEB_CONCURRENCY`.

Then open:

```text
http://127.0.0.1:8000/health
```

`/ready` additionally verifies that at least one model provider key is configured.

## Production configuration

- Keep `LOG_MODEL_RAW_OUTPUT=false` to avoid logging patient data.
- Set `CORS_ALLOWED_ORIGINS` to the frontend origins, separated by commas. Do not use `*` when credentials are required.
- Tune `MAX_UPLOAD_BYTES`, `MAX_PDF_PAGES`, `MAX_RENDERED_PIXELS`, `MAX_CONCURRENT_OCR`, and `MAX_OCR_SECONDS` for the deployment.
- Store API keys in the hosting platform's secret manager; never bake `.env` into an image.
- The Docker image runs as a non-root user, exposes a health check, and supports `PORT` and `WEB_CONCURRENCY`.

Run the regression checks with:

```powershell
..\venv\Scripts\python.exe -m unittest -v
```

## API Endpoint Reference

### GET `/health`
Returns `{"status": "ok"}` when healthy.

### GET `/ready`
Returns `{"status": "ready"}` when a provider is configured.

### POST `/ocr`
Accepts PDF, JPG, PNG, or WEBP file upload, processes it, and returns only the required prescription details. Invalid uploads use 4xx responses; exhausted model providers return 503 with a request ID.
- **Request**: Multipart Form Data with a `file` field containing the file.
- **Response**:
  ```json
  {
    "success": true,
    "Name": "Patient name",
    "Prescription": "Prescription number",
    "Address": "Patient address",
    "Mobile": "Mobile number",
    "Gender": "Gender",
    "Age": 45,
    "Medicines": [
      {
        "Name": "Medicine name",
        "Qty": 10
      }
    ]
  }
  ```
