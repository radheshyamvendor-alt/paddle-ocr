import os
os.environ['FLAGS_use_onednn'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
import io
import cv2
import fitz  # PyMuPDF
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR

app = FastAPI(title="Radheshyam PaddleOCR Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded PaddleOCR instance to avoid memory warnings/OOM at startup on Render
ocr_instance = None

def get_ocr():
    global ocr_instance
    if ocr_instance is None:
        ocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
    return ocr_instance

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Tailored medical document preprocessing:
    1. Grayscale Conversion
    2. Non-Local Means Denoising (noise reduction)
    3. CLAHE (Local Contrast Enhancement)
    4. 2D Filter Kernel Sharpening
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    filename = file.filename.lower() if file.filename else ""
    file_bytes = await file.read()
    
    combined_text = []
    
    try:
        if filename.endswith(".pdf") or file.content_type == "application/pdf":
            # PDF Processing: Render each page at 300 DPI to an image and run OCR
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid PDF: {str(e)}")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_data = pix.tobytes("png")
                
                try:
                    preprocessed = preprocess_image(img_data)
                    preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
                except Exception:
                    nparr = np.frombuffer(img_data, np.uint8)
                    preprocessed_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                result = get_ocr().ocr(preprocessed_bgr)
                if result and result[0]:
                    for line in result[0]:
                        box, (text, confidence) = line
                        combined_text.append(text)
        else:
            # Image Processing (JPG, JPEG, PNG, WEBP)
            try:
                preprocessed = preprocess_image(file_bytes)
                preprocessed_bgr = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
            except Exception:
                nparr = np.frombuffer(file_bytes, np.uint8)
                preprocessed_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if preprocessed_bgr is None:
                    raise HTTPException(status_code=400, detail="Invalid image format")
            
            result = get_ocr().ocr(preprocessed_bgr)
            if result and result[0]:
                for line in result[0]:
                    box, (text, confidence) = line
                    combined_text.append(text)
                    
        full_text = "\n".join(combined_text)
        return {
            "success": True,
            "text": full_text
        }
        
    except Exception as e:
        return {"success": False, "error": f"OCR extraction failed: {str(e)}"}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
