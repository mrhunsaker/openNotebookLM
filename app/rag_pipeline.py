import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple
import requests
import torch

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Global shared components
shared_embedding_model = None
shared_llm_model = None
chroma_client = None


async def initialize_rag_pipeline(granite_model_name: str, chroma_persist_dir: str):
    """
    Initializes the shared LLM, embedding model, and ChromaDB client using transformers.
    """
    global shared_llm_model, shared_embedding_model, chroma_client

    # Initialize embedding model
    try:
        shared_embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model (all-MiniLM-L6-v2) initialized.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        raise

    # Initialize Granite LLM using transformers
    try:
        if granite_model_name:
            print(f"Loading Granite LLM model: {granite_model_name}")

            # Check if ROCm/AMD GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name()}")
                print(f"ROCm version: {torch.version.hip}")
            else:
                print("Using CPU for inference")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                granite_model_name,
                trust_remote_code=True,
                padding_side="left"
            )

            # Add pad token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Configure model loading for ROCm
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }

            if device == "cuda":
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(
                granite_model_name,
                **model_kwargs
            )

            # Create text generation pipeline
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "return_full_text": False
            }

            if device == "cuda":
                pipeline_kwargs["device"] = 0
            else:
                pipeline_kwargs["device"] = -1

            text_generation_pipeline = pipeline(
                "text-generation",
                **pipeline_kwargs
            )

            # Wrap in LangChain HuggingFacePipeline
            shared_llm_model = HuggingFacePipeline(
                pipeline=text_generation_pipeline,
                model_kwargs={
                    "temperature": 0.7,
                    "max_new_tokens": 512,
                    "do_sample": True,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1
                }
            )

            print(f"Granite LLM loaded successfully using transformers on {device}.")
        else:
            print("Warning: No Granite model specified. Using a dummy LLM.")
            from langchain_community.llms import FakeListLLM
            shared_llm_model = FakeListLLM(responses=["I am a placeholder LLM response."])

        print("LLM model initialized.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

    # Initialize ChromaDB
    try:
        import chromadb
        chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        print(f"ChromaDB Persistent Client initialized at {chroma_persist_dir}.")
        os.makedirs(chroma_persist_dir, exist_ok=True)
    except Exception as e:
        print(f"Error initializing ChromaDB Client: {e}")
        raise

    return {
        "llm": shared_llm_model,
        "embedding_model": shared_embedding_model,
        "chroma_client": chroma_client
    }


async def get_or_create_vector_store_for_notebook(notebook_name: str, embedding_model, chroma_persist_dir: str) -> Chroma:
    """
    Gets or creates a ChromaDB collection for a given notebook name.
    """
    global chroma_client

    if chroma_client is None:
        raise ValueError("ChromaDB client not initialized. Call initialize_rag_pipeline first.")

    print(f"Accessing/creating ChromaDB collection for notebook: {notebook_name}")
    vector_store = Chroma(
        client=chroma_client,
        collection_name=notebook_name,
        embedding_function=embedding_model,
        persist_directory=chroma_persist_dir
    )
    print(f"ChromaDB vector store for '{notebook_name}' ready.")
    return vector_store


async def call_marker_ocr(file_path: str, marker_api_url: str) -> str:
    """
    Sends a file to the Marker OCR service and returns the text content.
    """
    print(f"Calling Marker OCR service at {marker_api_url} for file: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'application/octet-stream')}
            response = requests.post(f"{marker_api_url}/ocr", files=files, timeout=300)
            response.raise_for_status()
            ocr_response = response.json()
            if ocr_response.get("status") == "success":
                print(f"Marker OCR successful for {file_path}.")
                return ocr_response.get("text_content", "")
            else:
                raise Exception(f"Marker OCR service returned error: {ocr_response.get('message', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Marker OCR service: {e}. Falling back to empty string.")
        return ""
    except Exception as e:
        print(f"Unexpected error during Marker OCR call: {e}. Falling back to empty string.")
        return ""


async def call_docling_api(text_content: str, docling_api_url: str) -> List[Dict]:
    """
    Calls the Docling API to tokenize content and provide paragraph-level context.
    """
    print(f"Calling Docling API at {docling_api_url} for text content (first 100 chars): {text_content[:100]}...")
    headers = {"Content-Type": "application/json"}
    payload = {"text": text_content}
    try:
        response = requests.post(f"{docling_api_url}/parse", json=payload, headers=headers, timeout=600)
        response.raise_for_status()
        parsed_data = response.json()
        print("Docling API call successful.")

        documents_from_docling = []
        if isinstance(parsed_data, dict) and 'chunks' in parsed_data:
             for chunk_data in parsed_data['chunks']:
                 if 'text' in chunk_data:
                     metadata = chunk_data.get('metadata', {})
                     documents_from_docling.append({"text": chunk_data['text'], "metadata": metadata})
        elif isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, str):
                    documents_from_docling.append({"text": item, "metadata": {}})
                elif isinstance(item, dict) and 'text' in item:
                    documents_from_docling.append({"text": item['text'], "metadata": item.get('metadata', {})})
        else:
            print(f"Docling returned unexpected structure. Raw: {parsed_data}")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_text(text_content)
            documents_from_docling = [{"text": chunk, "metadata": {"source": "Docling_fallback_simple_split"}} for chunk in chunks]

        if not documents_from_docling and text_content:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = text_splitter.split_text(text_content)
            documents_from_docling = [{"text": chunk, "metadata": {"source": "Docling_fallback_empty_response"}} for chunk in chunks]

        return documents_from_docling

    except requests.exceptions.RequestException as e:
        print(f"Error calling Docling API: {e}. Falling back to simple text splitting.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = text_splitter.split_text(text_content)
        return [{"text": chunk, "metadata": {"source": "simple_splitter_fallback"}} for chunk in chunks]


async def process_document_folder(
    folder_path: str,
    vector_store: Chroma,
    embedding_model,
    docling_api_url: str,
    marker_api_url: str
):
    """
    Processes all files in a given folder.
    """
    print(f"Starting to process documents in: {folder_path} for collection: {vector_store._collection.name}")
    documents_to_add: List[Document] = []
    processed_files_count = 0

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_extension = Path(file_path).suffix.lower()
            text_content = ""
            file_metadata = {"source_filepath": file_path, "filename": file_name}

            try:
                # PDF handling: Try PyPDFLoader for accessible text, then Marker for OCR
                if file_extension == '.pdf':
                    try:
                        loader = PyPDFLoader(file_path)
                        pages = loader.load()
                        text_content = "\n".join(p.page_content for p in pages)
                        print(f"Loaded PDF text from {file_name} with PyPDFLoader.")
                        # If PyPDFLoader works, enrich metadata for pages
                        for i, p in enumerate(pages):
                            page_metadata = {**file_metadata, "page": i + 1}
                            # Docling will process the combined text, so we'll re-associate page numbers there.
                            # For now, just extract combined text.
                    except Exception as e:
                        print(f"PyPDFLoader failed for {file_name}: {e}. Attempting Marker OCR.")
                        text_content = await call_marker_ocr(file_path, marker_api_url)
                elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff']:
                    text_content = await call_marker_ocr(file_path, marker_api_url)
                elif file_extension == '.txt':
                    loader = TextLoader(file_path)
                    text_content = loader.load()[0].page_content
                elif file_extension == '.docx':
                    loader = Docx2txtLoader(file_path)
                    text_content = loader.load()[0].page_content
                else:
                    print(f"Skipping unsupported file type: {file_name}")
                    continue

                if text_content:
                    parsed_chunks_from_docling = await call_docling_api(text_content, docling_api_url)

                    for i, chunk_data in enumerate(parsed_chunks_from_docling):
                        chunk_text = chunk_data.get("text", "")
                        metadata = {**file_metadata, **chunk_data.get("metadata", {})}
                        metadata["chunk_id"] = i

                        if chunk_text:
                            documents_to_add.append(Document(page_content=chunk_text, metadata=metadata))
                    processed_files_count += 1
                else:
                    print(f"No content extracted or parsed from {file_name}.")

            except Exception as e:
                print(f"Failed to process {file_name}: {e}")

    if documents_to_add:
        print(f"Adding {len(documents_to_add)} document chunks to ChromaDB collection '{vector_store._collection.name}'...")
        vector_store.add_documents(documents=documents_to_add)
        print(f"Successfully processed {processed_files_count} files and added chunks to ChromaDB collection '{vector_store._collection.name}'.")
    else:
        print("No documents processed or added.")


async def chat_with_data(query: str, vector_store: Chroma, llm) -> Tuple[str, List[str]]:
    """
    Performs RAG to answer a user query using the specified vector_store (collection).
    """
    if not vector_store or not llm:
        raise ValueError("RAG components (vector store or LLM) not initialized.")

    print(f"Searching for relevant documents in collection '{vector_store._collection.name}' for query: '{query}'")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain({"query": query})
    response_text = result["result"]
    source_documents = result["source_documents"]

    citations = []
    for doc in source_documents:
        source_filepath = doc.metadata.get("source_filepath", "Unknown File")
        filename = doc.metadata.get("filename", os.path.basename(source_filepath))
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        page_num = doc.metadata.get("page", "N/A")

        citation_text = f"File: {filename}"
        if page_num != "N/A":
            citation_text += f" (Page: {page_num})"
        citation_text += f" (Chunk ID: {chunk_id})"

        citations.append(citation_text)

    return response_text, citations
