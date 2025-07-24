# marker_service/marker_api_server.py
import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn
import tempfile
from pathlib import Path
import json

# Add Marker's root directory to sys.path
sys.path.insert(0, os.path.abspath('.'))

try:
    from marker.models import load_all_models
    from marker.convert import convert_single_pdf
    print("Loading Marker models for API server...")
    # It's crucial to load models only once on server startup
    # Ensure they are loaded to GPU if available and configured
    models, tokenizer = load_all_models()
    print("Marker models loaded.")
except ImportError:
    print("Marker library not found. Marker OCR API will not function.")
    models, tokenizer = None, None
except Exception as e:
    print(f"Failed to load Marker models: {e}. Marker OCR API will not function correctly.")
    models, tokenizer = None, None

app = FastAPI()

class OCRResponse(BaseModel):
    text_content: str
    structured_json: dict = None
    status: str = "success"
    message: str = ""

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(file: UploadFile = File(...)):
    if models is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Marker models not loaded. OCR service is unavailable.")

    # Create a temporary file to save the uploaded PDF/image
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name

    try:
        output_dir = Path(tempfile.mkdtemp())
        pdf_name = Path(tmp_file_path).stem
        output_json_path = output_dir / f"{pdf_name}.json"

        print(f"Processing file: {tmp_file_path} with Marker...")
        # Marker's convert_single_pdf expects an output directory for JSONs
        # and it will return a list of paths if parallelized, or a single path.
        # Ensure parallelize=False for simpler single-file processing via API
        processed_docs = convert_single_pdf(
            tmp_file_path,
            output_dir, # Marker saves JSONs to this directory
            models=models,
            tokenizer=tokenizer,
            parallelize=False, # Important for API calls
            # Add other Marker options here if needed, e.g., max_pages, workers
            # workers=1 # Ensure single thread for API request
        )
        print(f"Marker processing complete. Output at: {output_json_path}")

        if output_json_path.exists():
            with open(output_json_path, 'r', encoding='utf-8') as f:
                marker_output = json.load(f)

            # Marker's JSON output might vary. It typically has 'text_content' and other structured data.
            text_content = marker_output.get("text_content", "")
            # You might want to parse 'pages' or other keys for structured_json
            structured_json = marker_output # Return the whole JSON for Docling to process

            return OCRResponse(
                text_content=text_content,
                structured_json=structured_json,
                status="success",
                message="File successfully OCR'd by Marker."
            )
        else:
            raise HTTPException(status_code=500, detail="Marker did not produce an output JSON file.")

    except Exception as e:
        print(f"Marker OCR error: {e}")
        raise HTTPException(status_code=500, detail=f"Marker OCR processing failed: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    port = int(os.getenv("MARKER_PORT", 8003))
    print(f"Starting Marker OCR API server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
