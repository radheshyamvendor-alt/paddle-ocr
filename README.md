# Radheshyam NVIDIA NIM Vision OCR Microservice

This is the NVIDIA NIM vision-based extraction service for the Radheshyam Medical Vendor application.

## Local Setup

Use Python 3.10 or 3.11 for this project.

Create a `.env` file from `.env.example` and set your NVIDIA API key:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```text
NVIDIA_API_KEY=your_real_key
NVIDIA_MODEL=meta/llama-3.2-11b-vision-instruct
```

From `E:\OCR\paddle-ocr`:

```powershell
..\venv\Scripts\python.exe -m ensurepip --upgrade
..\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
..\venv\Scripts\python.exe -m pip install -r requirements.txt
..\venv\Scripts\python.exe main.py
```

Then open:

```text
http://127.0.0.1:8000/health
```

## API Endpoint Reference

### GET `/health`
Returns `{"status": "ok"}` when healthy.

### POST `/ocr`
Accepts PDF, JPG, JPEG, PNG, or WEBP file upload, processes it, and returns only the required prescription details.
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
