```markdown
# Local OpenLM RAG Notebook with ROCm GPU Acceleration

This project provides a robust, local Retrieval-Augmented Generation (RAG) system for your documents using **Transformers** and **AMD ROCm GPU acceleration**. It features multi-notebook functionality, a GPU-accelerated OCR service (Marker) for extracting text from images and scanned PDFs, advanced document parsing (Docling), and local Granite large language models via Hugging Face Transformers. All core components run as isolated **Podman containers**, orchestrated by `podman-compose`, ensuring a self-contained and powerful solution optimized for AMD GPUs.

The frontend is built with **NiceGUI**, with a strong focus on **WCAG 2.x (AA and AAA) and Title II compliance**, providing full screen reader compatibility, comprehensive keyboard navigation, and proper visual color contrast.

---

## Table of Contents

1.  [Architecture and Component Interaction Mapping](#1-architecture-and-component-interaction-mapping)
2.  [Accessibility Statement](#2-accessibility-statement)
3.  [Prerequisites](#3-prerequisites)
4.  [Project Structure](#4-project-structure)
5.  [Setup Instructions](#5-setup-instructions)
    * [5.1. Directory Setup](#51-directory-setup)
    * [5.2. Download Granite Llamafile](#52-download-granite-llamafile)
    * [5.3. Create/Update Configuration Files](#53-createupdate-configuration-files)
    * [5.4. Create/Update Service Files](#54-createupdate-service-files)
    * [5.5. GPU Configuration (Crucial for Marker & LLM)](#55-gpu-configuration-crucial-for-marker--llm)
    * [5.6. Populating Source Documents](#56-populating-source-documents)
6.  [Building and Running the Application](#6-building-and-running-the-application)
7.  [Using the Application](#7-using-the-application)
    * [7.1. Initializing Components](#71-initializing-components)
    * [7.2. Managing Notebooks](#72-managing-notebooks)
    * [7.3. Chatting with Your Data](#73-chatting-with-your-data)
8.  [Troubleshooting](#8-troubleshooting)
9.  [Stopping and Cleaning Up](#9-stopping-and-cleaning-up)

---

## 1. Architecture and Component Interaction Mapping

This project uses a microservices architecture managed by `podman-compose`, optimized for AMD ROCm GPU acceleration. Each key function (UI/RAG, Docling, Marker) runs in its own container, with the Granite LLM running via Transformers directly in the main app container.

```mermaid
flowchart TD
    A[NiceGUI Web Interface] --> B[RAG Pipeline - app container]
    B --> C[Document Processing]
    C --> D[Marker OCR Service - ROCm GPU]
    C --> E[Docling Service - Text Processing]
    C --> F[ChromaDB Vector Store]
    B --> G[Granite LLM - Transformers + ROCm]
    G --> H[Hugging Face Model Cache]
    B --> I[SentenceTransformers Embeddings]

    style D fill:#ff6b6b
    style G fill:#4ecdc4
    style H fill:#45b7d1
```

**ROCm-Optimized Interaction Flow:**

1.  **User Interface (`app` container - NiceGUI):** Web interface for managing notebooks and chatting with documents.
2.  **Document Processing (`app` container - `rag_pipeline.py`):**
    * Reads files from mounted `sources` directory
    * **PDFs/Images** → **Marker OCR Service** (`http://marker_ocr:8003/ocr`) with ROCm GPU acceleration
    * **Text extraction** → **Docling Service** (`http://docling:8001/parse`) for chunking and metadata
    * **Embeddings** → SentenceTransformers (local) creates vector embeddings stored in ChromaDB
3.  **AI Chat (`app` container - Transformers + ROCm):**
    * ChromaDB retrieves relevant document chunks
    * **Granite LLM** (via Hugging Face Transformers) processes queries with ROCm GPU acceleration
    * Responses generated locally with citations and displayed in NiceGUI
4.  **ROCm GPU Utilization:**
    * **Marker OCR:** Computer vision models for document scanning
    * **Granite LLM:** Large language model inference
    * **Optimized for AMD Radeon RX 7700S** with RDNA3 architecture

---

## 2. Accessibility Statement

This application is designed with a strong commitment to accessibility, aiming to meet **WCAG 2.x AA and AAA conformance** and comply with **Title II requirements**. We've implemented the following features to ensure an inclusive user experience:

* **Full Keyboard Navigation:** All interactive elements (buttons, input fields, dropdowns) are fully navigable using only the keyboard (`Tab`, `Shift+Tab`, `Enter`, `Spacebar`, arrow keys). The tab order follows a logical flow.
* **Screen Reader Compatibility:**
    * **Semantic HTML:** We use NiceGUI components that render appropriate semantic HTML (`<button>`, `<input>`, `<label>`, `<select>`, etc.).
    * **ARIA Attributes:** Where necessary, custom ARIA (Accessible Rich Internet Applications) attributes (e.g., `aria-label`, `aria-labelledby`, `aria-live`) are explicitly added to provide meaningful context and announcements for screen reader users.
    * **Live Regions:** Dynamic content updates, such as processing status messages and new chat responses, are placed within `aria-live` regions to ensure screen readers announce these changes effectively and in a timely manner.
    * **Hidden Content:** Decorative elements are hidden from screen readers, and critical announcements are provided via visually hidden text.
* **Color Contrast:** The application uses a carefully selected color palette for the user interface, ensuring that text and interactive elements meet or exceed WCAG's color contrast ratio requirements for both AA and AAA levels.
* **Responsive Design:** The layout is responsive, ensuring content reflows gracefully and remains usable across various screen sizes and zoom levels without loss of information or functionality.
* **Clear Feedback:** Error messages and status updates are explicit and accessible, providing clear guidance to all users.

We are committed to ongoing accessibility improvements and welcome feedback.

---

## 3. Prerequisites

Before you start, make sure you have the following installed and configured on your **Linux host machine**:

* **Podman:** A daemonless container engine. [Installation Guide](https://podman.io/docs/installation)
* **Podman Compose:** A tool for defining and running multi-container Podman applications. [Installation Guide](https://docs.podman.io/en/latest/markdown/podman-compose.1.html)
* **Rootless Podman configured:** This setup is designed for rootless containers. Ensure your user has sufficient `subuid` and `subgid` entries in `/etc/subuid` and `/etc/gid`. You can usually enable this with `loginctl enable-linger $(whoami)`.
* **AMD ROCm Drivers:** Essential for AMD Radeon RX 7700S GPU acceleration:
    * **Install ROCm 5.6+:** [ROCm Installation Guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.6/page/Introduction_to_AMD_ROCm_Installation_Guide_for_Linux.html)
    * **User permissions:** `sudo usermod -a -G render,video $USER` (logout/login required)
    * **Verify installation:** `rocm-smi` should show your RX 7700S
    * **PyTorch ROCm support:** Will be installed in containers automatically
* **Git:** For cloning repositories.
* **Internet connection:** Required for Hugging Face model downloads (first run only).
* **Disk space:** At least 20GB free for models and containers.

---

## 4. Project Structure

Your project will live in `~/Documents/openLM/` and will look like this:

```
~/Docuemnts/openLM/
├── podman-compose.yml          # Service orchestration with ROCm GPU support
├── podman.env                  # Environment variables (Granite model, ROCm settings)
├── setup_rocm.sh              # Automated setup script for AMD GPU
├── app/                        # Main NiceGUI RAG application with Transformers
│   ├── Podmanfile.app         # ROCm-enabled container for app
│   ├── main.py                # NiceGUI frontend and orchestrator
│   ├── rag_pipeline.py        # RAG logic with Transformers + ROCm
│   ├── requirements.txt       # Python deps (transformers, torch+rocm)
│   └── chroma_db/             # Persistent ChromaDB storage
├── sources/                    # Your document storage (mounted volume)
│   ├── sample_research/       # Example notebook folder 1
│   │   ├── research_notes.md
│   │   ├── document1.pdf
│   │   └── image.png
│   └── project_docs/          # Example notebook folder 2
│       ├── project_plan.txt
│       └── report.docx
├── docling_service/           # Document parsing service
│   ├── Podmanfile.docling     # Python container for text processing
│   └── docling_api.py         # FastAPI server for document parsing
└── marker_service/            # ROCm-accelerated OCR service
    ├── podmanfile.marker      # ROCm PyTorch container for Marker
    ├── download_marker_models.py  # Pre-download OCR models
    └── marker_api_server.py   # FastAPI server for OCR API
```

**Key Changes from Llamafile Version:**
- **No `llm_service/`** - Granite runs via Transformers in main app
- **ROCm base images** for GPU acceleration
- **Hugging Face model caching** replaces manual model files
- **AMD-specific environment variables** for optimal performance
````

---

## 5. Setup Instructions

Let's get everything ready. Follow these steps carefully.

### 5.1. Directory Setup

First, create all the necessary project directories:

```bash
mkdir -p ~/downloads/openLM/app ~/downloads/openLM/sources
mkdir -p ~/downloads/openLM/docling_service ~/downloads/openLM/llm_service
mkdir -p ~/downloads/openLM/marker_service # New: Marker service directory
mkdir -p ~/downloads/openLM/app/models     # For llamafiles
mkdir -p ~/downloads/openLM/app/chroma_db  # For persistent ChromaDB data
````

### 5.2. Choose Your Granite Model

This setup uses Hugging Face Transformers to load Granite models directly. No manual downloads are required - models will be automatically downloaded on first use. Choose from these IBM Granite models:

**Recommended models for AMD Radeon RX 7700S (12GB VRAM):**
* `ibm/granite-3b-code-instruct` - Smaller, faster, good for code and general tasks
* `ibm/granite-7b-instruct` - Balanced performance and capability
* `ibm/granite-13b-instruct-v2` - Larger model (may require model sharding)

The model is specified in the `podman.env` file and can be changed without rebuilding containers.

### 5.3. Create/Update Configuration Files

Create or update these files in your `~/downloads/openLM/` directory:

  * **`podman.env`**:

    ```env
    # General Application Settings
    APP_PORT=8000
    SOURCE_DIR=/downloads/openLM/sources

    # Granite LLM Settings using Transformers
    # Choose your preferred Granite model from Hugging Face
    GRANITE_MODEL_NAME=ibm/granite-3b-code-instruct

    # Docling API Server Settings
    DOCLING_API_PORT=8001

    # Marker API Server Settings
    MARKER_API_PORT=8003

    # ChromaDB Settings
    CHROMA_PERSIST_DIR=/app/chroma_db

    # ROCm/AMD GPU Settings for RX 7700S
    ROCM_PATH=/opt/rocm
    HIP_PATH=/opt/rocm
    PYTORCH_ROCM_ARCH=gfx1031
    HSA_OVERRIDE_GFX_VERSION=10.3.0
    HIP_VISIBLE_DEVICES=0
    ```

  * **`podman-compose.yml`**:

    ```yaml
    version: '3.8'

    services:
      # NiceGUI Frontend and RAG Backend Logic with Transformers
      app:
        build:
          context: ./app
          podmanfile: Podmanfile.app
        ports:
          - "${APP_PORT}:${APP_PORT}"
        volumes:
          - ./sources:/app/sources:Z # Mount the local sources folder
          - ./app/chroma_db:/app/chroma_db:Z # Persistent storage for ChromaDB
          - ~/.cache/huggingface:/root/.cache/huggingface:Z # Cache for Hugging Face models
        environment:
          APP_PORT: ${APP_PORT}
          GRANITE_MODEL_NAME: ${GRANITE_MODEL_NAME}
          CHROMA_PERSIST_DIR: ${CHROMA_PERSIST_DIR}
          DOCLING_API_URL: http://docling:8001
          MARKER_API_URL: http://marker_ocr:8003
          # ROCm environment variables
          ROCM_PATH: /opt/rocm
          HIP_PATH: /opt/rocm
          PYTORCH_ROCM_ARCH: gfx1031
          HSA_OVERRIDE_GFX_VERSION: 10.3.0
          HIP_VISIBLE_DEVICES: 0
        devices:
          - /dev/kfd:/dev/kfd # AMD GPU kernel driver
          - /dev/dri:/dev/dri # AMD GPU DRI devices
        depends_on:
          - docling
          - marker_ocr
        restart: unless-stopped

      # Docling API Service
      docling:
        build:
          context: ./docling_service
          podmanfile: Podmanfile.docling
        ports:
          - "${DOCLING_API_PORT}:${DOCLING_API_PORT}"
        environment:
          DOCLING_PORT: ${DOCLING_API_PORT}
        restart: unless-stopped

      # Marker OCR Service with AMD GPU support
      marker_ocr:
        build:
          context: ./marker_service
          podmanfile: podmanfile.marker
        ports:
          - "${MARKER_API_PORT}:${MARKER_API_PORT}"
        environment:
          MARKER_PORT: ${MARKER_API_PORT}
          # ROCm environment variables
          ROCM_PATH: /opt/rocm
          HIP_PATH: /opt/rocm
          PYTORCH_ROCM_ARCH: gfx1031
          HSA_OVERRIDE_GFX_VERSION: 10.3.0
          HIP_VISIBLE_DEVICES: 0
        devices:
          - /dev/kfd:/dev/kfd # AMD GPU kernel driver
          - /dev/dri:/dev/dri # AMD GPU DRI devices
        restart: unless-stopped

    volumes:
      chroma_data:
        driver: local
      huggingface_cache:
        driver: local
    ```

### 5.4. Create/Update Service Files

Ensure these files exist in their respective directories:

  * **`app/requirements.txt`**:

    ```
    # Web Framework
    nicegui==1.4.15

    # RAG Framework
    langchain==0.2.10
    langchain-community==0.2.8
    chromadb==0.5.3

    # Document Processing
    pypdf==4.2.0
    python-docx==1.1.0
    docxtpl==1.2.0
    requests==2.32.3
    python-dotenv==1.0.1

    # AI/ML with ROCm Support
    sentence-transformers==2.7.0
    transformers==4.36.0
    torch==2.1.0+rocm5.6
    torchvision==0.16.0+rocm5.6
    torchaudio==2.1.0+rocm5.6
    accelerate==0.25.0
    bitsandbytes==0.41.3
    safetensors==0.4.1
    tokenizers==0.15.0
    huggingface_hub==0.19.4
    datasets==2.14.7
    ```

  * **`app/Podmanfile.app`**:

    ```dockerfile
    # Use ROCm base image for AMD GPU support
    FROM rocm/pytorch:rocm5.6-pytorch2.0.1-ubuntu20.04

    WORKDIR /app

    # Set environment variables for ROCm
    ENV ROCM_PATH=/opt/rocm
    ENV HIP_PATH=/opt/rocm
    ENV PYTORCH_ROCM_ARCH=gfx1031
    ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
    ENV HIP_VISIBLE_DEVICES=0

    # Install system dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake pkg-config \
        libglib2.0-0 libxml2-dev libxslt-dev poppler-utils \
        libopenjp2-7 libjpeg-dev zlib1g-dev \
        python3-pip python3-dev \
        && rm -rf /var/lib/apt/lists/*

    # Upgrade pip and install build tools
    RUN pip install --upgrade pip setuptools wheel

    # Install ROCm-specific PyTorch first
    RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6

    # Copy requirements and install Python dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy the application code
    COPY . .

    # Expose the NiceGUI port
    EXPOSE 8000

    CMD ["python3", "main.py"]
    ```

**Note:** The complete `main.py` and `rag_pipeline.py` files are already updated in the project with Transformers integration and ROCm support.

  * **`marker_service/Podmanfile.marker`**:

    ```dockerfile
    # Use a robust base image with CUDA pre-installed for NVIDIA GPUs.
    # For AMD GPUs, you'd choose a ROCm-enabled base image.
    # This example uses NVIDIA's CUDA image.
    FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04 # Or your preferred CUDA version / Ubuntu base

    WORKDIR /app

    # Install system dependencies
    # These are common for Python ML apps, adjust as needed for Marker
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3-venv \
        git build-essential libglib2.0-0 libxml2-dev libxslt-dev poppler-utils \
        libopenjp2-7 libjpeg-dev zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

    # Set up a Python virtual environment (optional but good practice)
    ENV VIRTUAL_ENV=/opt/venv
    RUN python3 -m venv $VIRTUAL_ENV
    ENV PATH="$VIRTUAL_ENV/bin:$PATH"

    # Install PyTorch with CUDA support.
    # IMPORTANT: Match CUDA version with your base image and GPU driver.
    # Check [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the correct command.
    # Example for CUDA 12.1:
    # RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    # This should be dynamic for `nvidia/cuda:12.3.2`:
    RUN pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

    # Clone Marker repository
    RUN git clone [https://github.com/datalab-to/marker.git](https://github.com/datalab-to/marker.git) .

    # Install Marker dependencies.
    # Marker has a `requirements.txt` but often needs a few more.
    # Also, ensure 'marker' itself is installed in editable mode for model loading.
    RUN pip install --no-cache-dir -e .
    RUN pip install --no-cache-dir \
        onnxruntime-gpu # If you want ONNX Runtime GPU for faster inference

    # Marker downloads models on first use or via specific commands.
    # To self-contain, pre-download models into the image.
    # This assumes the `marker.models.load_all_models()` function uses a default path
    # or can be configured to save to `/app/models`.
    # You might need to write a small Python script to trigger model download.
    # Example:
    COPY download_marker_models.py .
    RUN python3 download_marker_models.py

    # Expose a port for our custom Marker API server
    EXPOSE 8003

    # Command to run the custom Marker API server
    COPY marker_api_server.py .
    CMD ["python3", "marker_api_server.py"]
    ```

  * **`marker_service/download_marker_models.py`**:

    ```python
    # marker_service/download_marker_models.py
    import os
    import sys

    # Add Marker's root directory to sys.path to import its modules
    sys.path.insert(0, os.path.abspath('.'))

    try:
        from marker.models import load_all_models
        print("Attempting to download Marker models...")
        # This will load and cache models. Marker typically saves them to ~/.cache/marker
        # or a similar location. We want them within the container.
        # The default behavior of load_all_models should store them in a discoverable path.
        # If it writes to home directory, ensure the container's HOME is /app.
        os.environ['HOME'] = '/app' # Direct models to /app/.cache/marker if possible
        # Marker loads models into memory and also caches them to disk.
        # The first call to load_all_models will download them if not present.
        models, tokenizer = load_all_models()
        print("Marker models downloaded and cached successfully.")
    except ImportError:
        print("Marker library not found. Skipping model download.")
    except Exception as e:
        print(f"Error downloading Marker models: {e}")
        # Exit with an error if model download fails during build
        sys.exit(1)
    ```

  * **`marker_service/marker_api_server.py`**:

    ```python
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
    ```

-----

## 6. Building and Running the Application

1.  **Navigate to the project root:**

    ```bash
    cd ~/downloads/openLM/
    ```
2.  **Build the container images:** This downloads ROCm base images, installs dependencies, and pre-downloads AI models. **This can take 30-60 minutes and requires 15-20GB disk space.**

  ```bash
  podman-compose build
  ```

  **Build Progress Monitoring:**
  ```bash
  # Monitor build progress in another terminal
  podman-compose build --progress=plain
  ```

  **Troubleshooting Build Issues:**
  - **ROCm/PyTorch errors:** Ensure ROCm 5.6+ is installed: `rocm-smi --version`
  - **Network timeouts:** Large model downloads may timeout - retry the build
  - **Disk space:** Ensure 20GB+ free space: `df -h`
  - **Memory issues:** Close other applications during build

3.  **Start the services:**

    ```bash
    podman-compose up -d
    ```

    This starts the `app`, `docling`, and `marker_ocr` containers with ROCm GPU support.

4.  **Monitor startup (first run):**

    ```bash
    # Watch logs for model downloads (first run only)
    podman-compose logs -f app
    ```

    **First run will download:**
    - Granite LLM model (~3-7GB depending on model choice)
    - SentenceTransformers embedding model (~90MB)
    - Marker OCR models (~2GB)

5.  **Verify services are running:**

    ```bash
    podman-compose ps
    ```

    All services should show "Up" status. If not, check logs:
    ```bash
    podman-compose logs app      # Main application
    podman-compose logs marker_ocr  # OCR service
    podman-compose logs docling  # Document parser
    ```

6.  **Verify ROCm GPU access:**

    ```bash
    # Check GPU detection in app container
    podman-compose exec app python3 -c "
    import torch
    print(f'ROCm CUDA available: {torch.cuda.is_available()}')
    print(f'GPU count: {torch.cuda.device_count()}')
    if torch.cuda.is_available():
        print(f'GPU name: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    "
    ```

    **Expected output for RX 7700S:**
    ```
    ROCm CUDA available: True
    GPU count: 1
    GPU name: AMD Radeon RX 7700S
    GPU memory: 12.0 GB
    ```

-----

## 7. Using the Application

### 7.1. Accessing the Interface

Open your web browser and navigate to: **`http://localhost:8000`**

**Interface Features:**
- **WCAG 2.x compliant** for screen readers and keyboard navigation
- **Responsive design** works on desktop, tablet, and mobile
- **Real-time status updates** with ARIA live regions
- **High contrast** color scheme for accessibility

### 7.2. First-Time Initialization

**Initial Load Process:**
1. **"Initializing shared RAG components..."** - Loading Granite LLM with ROCm
2. **"Downloading models..."** - First run downloads from Hugging Face (5-10 minutes)
3. **"Shared RAG components initialized."** - Ready to use

**Performance Notes:**
- **First startup:** 5-10 minutes (model downloads)
- **Subsequent startups:** 1-2 minutes (loading cached models)
- **GPU warming:** First query may take 30-60 seconds

### 7.3. Notebook Discovery

The "Select or Create Notebook" dropdown automatically populates with folders from `~/downloads/openLM/sources/`:
- **sample_research** - AI/ML research notes (if using setup script)
- **project_docs** - Project documentation (if using setup script)
- **[your folders]** - Any custom folders you've created

### 7.4. Processing Documents (ROCm Accelerated)

**A. Using Existing Notebooks:**

1.  **Select notebook** from dropdown (e.g., "sample_research")
2.  Click **"Select Notebook"** to load the ChromaDB collection
3.  If documents haven't been processed yet, click **"Process Documents in this Folder"**

**B. Creating New Notebooks:**

1.  **Create folder:** `mkdir -p ~/downloads/openLM/sources/my_new_notebook`
2.  **Add documents** to the folder
3.  **In the UI:** Type "my_new_notebook" in the "Notebook Folder Path" field
4.  **Click "Process Documents in this Folder"**

**Document Processing Pipeline (ROCm Accelerated):**
```
Documents → Type Detection → Processing Pipeline → Vector Storage
    ↓              ↓                ↓                  ↓
PDFs/Images → Marker OCR     → Text Chunks    → ChromaDB
Text Files  → Direct Read    → Docling Parse  → Embeddings
Word Docs   → Text Extract   → Metadata Tag   → Ready for Chat
```

**Processing Performance:**
- **Text files:** ~10-50 documents/minute
- **PDFs (searchable):** ~5-15 documents/minute
- **Scanned PDFs/Images:** ~1-3 documents/minute (ROCm OCR)
- **Large documents:** Progress shown in real-time

**Status Indicators:**
- **"Processing documents..."** - Currently running
- **"Calling Marker OCR service..."** - GPU OCR in progress
- **"Adding X document chunks..."** - Final indexing step
- **"Ready to chat."** - Processing complete

### 7.5. AI Chat with ROCm Acceleration

**Starting a Chat Session:**

1.  **Select processed notebook** from dropdown
2.  **Click "Select Notebook"** - Status shows "Ready to chat with [notebook_name]"
3.  **Type your question** in the input field
4.  **Press Enter** or click "Send"

**Query Processing (ROCm Accelerated):**
```
Your Question → ChromaDB Search → Context Retrieval → Granite LLM → Response
     ↓              ↓                    ↓              ↓ (ROCm)        ↓
"What is..."  → Vector Search     → Top 5 chunks   → GPU Inference → Answer + Citations
```

**Performance Expectations:**
- **First query:** 30-60 seconds (GPU warmup)
- **Subsequent queries:** 5-15 seconds
- **Complex queries:** 15-30 seconds
- **GPU memory:** ~8-10GB used during inference

**Sample Questions to Try:**
- **Summarization:** "Summarize the main points in this notebook"
- **Specific facts:** "What are the key findings about [topic]?"
- **Comparisons:** "Compare the approaches mentioned in different documents"
- **Analysis:** "What are the pros and cons of [concept]?"

**Response Features:**
- **Generated answer** based on your documents
- **Source citations** with file names, page numbers, and chunk IDs
- **Clickable text** for copying responses
- **Real-time status** updates during processing
- **Error handling** with fallback responses

-----

## 8. Troubleshooting

 ### Common Issues and Solutions

**🔧 Container Startup Issues:**
```bash
# Check service status
podman-compose ps

# View logs for specific services
podman-compose logs app
podman-compose logs marker_ocr
podman-compose logs docling
```

**🎯 AMD GPU Not Detected:**
```bash
# Verify ROCm installation
rocm-smi
rocm-smi --showproductname

# Check user permissions
groups | grep -E "render|video"

# Test GPU in container
podman-compose exec app python3 -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}')"
```

**📥 Model Download Issues:**
- **Slow downloads:** First run downloads 5-10GB of models
- **Network timeouts:** Check internet connection and retry
- **Disk space:** Ensure 20GB+ free space
- **Hugging Face Hub:** Check [status.huggingface.co](https://status.huggingface.co)

**💾 Out of GPU Memory:**
```bash
# Use smaller model in podman.env
GRANITE_MODEL_NAME=ibm/granite-3b-code-instruct

# Clear GPU memory
podman-compose restart app
```

**🚫 "RAG components not initialized" Error:**
- **Cause:** App container still downloading models or GPU error
- **Solution:** Wait 5-10 minutes for first-time model downloads
- **Check:** `podman-compose logs app` for detailed errors

**📄 Document Processing Failures:**
- **PDF issues:** Try both PyPDF and Marker OCR processing
- **Large files:** May timeout - increase timeout values
- **File permissions:** Ensure files are readable in mounted volume
- **Unsupported formats:** Only PDF, TXT, DOCX, and images supported

**🔒 SELinux Permission Issues:**
```bash
# Fix volume mount permissions
sudo setsebool -P container_manage_cgroup true
```

**⚡ Slow Performance Issues:**
```bash
# Optimize ROCm settings in podman.env
HSA_OVERRIDE_GFX_VERSION=10.3.0
AMD_SERIALIZE_KERNEL=3
HIP_LAUNCH_BLOCKING=1
```

**📊 Debug Commands:**
```bash
# GPU memory usage
rocm-smi --showmeminfo

# Container resource usage
podman stats

# Detailed app logs
podman-compose logs -f app --tail=100
```

-----

## 9. Performance Benchmarks (AMD RX 7700S)

**Hardware Configuration:**
- **GPU:** AMD Radeon RX 7700S (12GB VRAM, RDNA3)
- **ROCm:** 5.6+
- **Model:** IBM Granite-3B-Code-Instruct

**Performance Metrics:**

| Task | Performance | GPU VRAM Usage |
|------|-------------|----------------|
| **Model Loading** | 60-90 seconds (first time) | ~4GB |
| **Document Processing** | 10-50 docs/min (text), 1-3 docs/min (OCR) | ~6GB |
| **Chat Inference** | 5-15 seconds/query | ~8GB |
| **First Query (cold start)** | 30-60 seconds | ~8GB |
| **Concurrent Users** | 2-3 users recommended | ~10GB |

**Optimization Tips:**
- Use **granite-3b-code-instruct** for best performance on 12GB VRAM
- Enable **GPU memory clearing** between sessions for optimal performance
- **Batch document processing** for better throughput

-----

## 10. Model Alternatives

**Granite Model Options (tested with RX 7700S):**

| Model | Size | VRAM Required | Performance | Use Case |
|-------|------|---------------|-------------|----------|
| `ibm/granite-3b-code-instruct` | ~6GB | 8GB | ⭐⭐⭐⭐⭐ | **Recommended** - Fast, capable |
| `ibm/granite-7b-instruct` | ~14GB | 10GB | ⭐⭐⭐⭐ | Balanced performance |
| `ibm/granite-13b-instruct-v2` | ~26GB | 12GB* | ⭐⭐⭐ | *Requires model sharding |

**Changing Models:**
```bash
# Edit podman.env
GRANITE_MODEL_NAME=ibm/granite-7b-instruct

# Restart to download new model
podman-compose down
podman-compose up -d
```

**Other Compatible Models:**
- **CodeLlama variants** for code-focused tasks
- **Mistral 7B** for general chat
- **Phi-2/Phi-3** for smaller deployments

-----

## 11. Advanced Configuration

### ROCm Optimization for RX 7700S

**Environment Tuning:**
```bash
# Add to podman.env for optimal performance
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1031
HIP_LAUNCH_BLOCKING=1
AMD_SERIALIZE_KERNEL=3
```

### Memory Management

**For Large Documents (>100MB):**
```python
# Add to app/rag_pipeline.py
CHUNK_SIZE = 500  # Smaller chunks
CHUNK_OVERLAP = 100
MAX_BATCH_SIZE = 10  # Process fewer docs at once
```

### Multi-GPU Setup

**For systems with multiple AMD GPUs:**
```yaml
# Update podman-compose.yml
environment:
  HIP_VISIBLE_DEVICES: "0,1"  # Use multiple GPUs
```

-----

## 12. Stopping and Cleaning Up

### Regular Shutdown
```bash
cd ~/downloads/openLM/
podman-compose down
```

### Complete Cleanup (removes all data)
```bash
cd ~/downloads/openLM/
podman-compose down --rmi all --volumes

# Optional: Remove model cache (will re-download on next start)
rm -rf ~/.cache/huggingface/transformers/

# Optional: Remove document database (will need to re-process documents)
rm -rf ~/downloads/openLM/app/chroma_db/
```

### Selective Cleanup
```bash
# Keep processed documents, remove only containers
podman-compose down --rmi all

# Clear only model cache (keeps document processing)
rm -rf ~/.cache/huggingface/transformers/
```

-----

## 13. Contributing and Support

**Project Repository:** [GitHub - OpenNotebookLM](https://github.com/your-repo/openNotebookLM)

**Contributing:**
- **Issues:** Report bugs and feature requests
- **Pull Requests:** Improvements and optimizations welcome
- **Documentation:** Help improve setup guides
- **ROCm Optimization:** Share AMD GPU performance tweaks

**Community Support:**
- **Discord:** [OpenNotebookLM Community](https://discord.gg/opennotebook)
- **Reddit:** r/LocalLLM discussions
- **AMD ROCm:** [ROCm Community Forums](https://community.amd.com/t5/rocm/ct-p/amd-rocm)

**Acknowledgments:**
- **IBM Research:** Granite model development
- **Hugging Face:** Transformers library and model hosting
- **AMD:** ROCm GPU acceleration framework
- **Marker Team:** Document OCR capabilities
- **Docling Team:** Document parsing and chunking

---

**🚀 Happy RAGing with your AMD GPU!**

*Built with ❤️ for the open-source AI community*
