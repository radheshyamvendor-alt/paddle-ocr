# Radheshyam PaddleOCR Microservice

This is the PaddleOCR-based extraction service for the Radheshyam Medical Vendor application.

## API Endpoint Reference

### GET `/health`
Returns `{"status": "ok"}` when healthy.

### POST `/ocr`
Accepts PDF, JPG, JPEG, PNG, or WEBP file upload, processes it, and returns the combined text.
- **Request**: Multipart Form Data with a `file` field containing the file.
- **Response**:
  ```json
  {
    "success": true,
    "text": "Full extracted text content..."
  }
  ```
