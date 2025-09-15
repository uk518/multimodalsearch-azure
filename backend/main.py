from azure.cognitiveservices.vision.computervision import ComputerVisionClient
import re
import json
import os
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
# Example usage (call these in your indexing/search logic):
# docs = [{"id": "1", "content": "Sample chunk", "embedding": [0.1, 0.2], "type": "text"}]
# azure_search_upload_documents(docs)
# search_results = azure_search_query("sample")
# print(search_results)
from indexing.azure_search import search_text_embedding, search_image_embedding
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
from extraction.pdf import extract_text_and_images
from extraction.image import extract_image_and_ocr
from embeddings.text import get_text_embedding
from embeddings.image import get_image_embedding
from indexing.azure_search import index_document
import requests
from urllib.parse import urlparse
from extraction.web import extract_from_url



app = FastAPI()

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def split_text(text, chunk_size=500):
    # Simple text splitter by chunk size
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def save_image_from_pdf(image_dict, save_dir, idx):
    import base64
    img_data = image_dict.get('data')
    if img_data:
        img_bytes = base64.b64decode(img_data)
        img_path = os.path.join(save_dir, f"pdf_image_{idx}.png")
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        return img_path
    return None


def sanitize_key(key):
    # Replace any character not allowed with underscore
    return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)



@app.post("/upload/pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    processed = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"Saved PDF: {file.filename}")
        # Extraction
        try:
            text, images = extract_text_and_images(file_path)
        except Exception as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=400, content={"error": f"PDF parsing failed: {str(e)}"})
        # Text preprocessing
        
        text_chunks = split_text(text)
        for i, chunk in enumerate(text_chunks):
            print(f"PDF chunk {i+1}: {chunk[:100]}...")
            embedding = get_text_embedding(chunk)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            doc_id = sanitize_key(f"pdf_{file.filename}_{i+1}")
            print(f"Indexing PDF chunk {i+1} with embedding shape: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
            index_document({"id": doc_id, "type": "text", "content": chunk, "embedding": embedding})
       
        # Image preprocessing
        for idx, img_dict in enumerate(images):
            img_path = save_image_from_pdf(img_dict, UPLOAD_DIR, idx)
            if img_path:
                meta = extract_image_and_ocr(img_path)
                meta['image_url'] = f"/static/{os.path.basename(img_path)}"
                embedding = get_image_embedding(img_path)
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                doc_id = sanitize_key(f"pdfimg_{file.filename}_{idx+1}")
                index_document({"id": doc_id, "type": "image", "meta": meta, "embedding": embedding})
        processed.append(file.filename)
    return {"status": "PDFs processed", "files": processed}


# @app.post("/upload/image")
# async def upload_image(files: List[UploadFile] = File(...)):
#     processed = []
#     for file in files:
#         file_path = os.path.join(UPLOAD_DIR, file.filename)
#         with open(file_path, "wb") as f:
#             f.write(await file.read())
#         print(f"Saved image: {file.filename}")
#         # Preprocess image
#         meta = extract_image_and_ocr(file_path)
#         print(f"Image metadata: {meta}")
#         embedding = get_image_embedding(file_path)
#         if hasattr(embedding, 'tolist'):
#             embedding = embedding.tolist()
#         print(f"Indexing image {file.filename} with embedding shape: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
#     doc_id = sanitize_key(f"img_{file.filename}")
#     index_document({"id": doc_id, "type": "image", "meta": meta, "embedding": embedding})
    
#     # OCR text embedding
#     ocr_text = meta.get('ocr_text') if isinstance(meta, dict) else None
#     if ocr_text:
#         print(f"Image OCR text: {ocr_text[:100]}...")
#         text_embedding = get_text_embedding(ocr_text)
#         if hasattr(text_embedding, 'tolist'):
#             text_embedding = text_embedding.tolist()
#         print(f"Indexing image OCR text with embedding shape: {len(text_embedding) if hasattr(text_embedding, '__len__') else 'unknown'}")
#         doc_id = sanitize_key(f"imgocr_{file.filename}")
#         index_document({"id": doc_id, "type": "text", "content": ocr_text, "embedding": text_embedding})
#     processed.append(file.filename)
#     return {"status": "Images processed", "files": processed}

# Load credentials from environment variables
AZURE_CV_ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
AZURE_CV_KEY = os.getenv("AZURE_CV_KEY")

print("AZURE_CV_KEY:", AZURE_CV_KEY)
print("AZURE_CV_ENDPOINT:", AZURE_CV_ENDPOINT)


# Initialize the Computer Vision client
computervision_client = ComputerVisionClient(
    AZURE_CV_ENDPOINT,
    CognitiveServicesCredentials(AZURE_CV_KEY)
)

def extract_image_and_ocr_azure(image_path):
    with open(image_path, "rb") as image_stream:
        ocr_result = computervision_client.recognize_printed_text_in_stream(image=image_stream)
        lines = []
        for region in ocr_result.regions:
            for line in region.lines:
                lines.append(" ".join([word.text for word in line.words]))
        return {"ocr_text": "\n".join(lines)}
    







@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    upload_dir = os.path.join(os.getcwd(), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    meta = extract_image_and_ocr_azure(file_path)
    meta['image_url'] = f"/static/{os.path.basename(file_path)}"
    embedding = get_image_embedding(file_path)
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    doc_id = sanitize_key(f"img_{file.filename}")
    import json
    index_document({"id": doc_id, "type": "image", "meta": json.dumps(meta), "embedding": embedding})
    return {"status": "success", "filename": file.filename, "ocr_text": meta['ocr_text']}
    
    # OCR text embedding
    ocr_text = meta.get('ocr_text') if isinstance(meta, dict) else None
    if ocr_text:
        print(f"Image OCR text: {ocr_text[:100]}...")
        text_embedding = get_text_embedding(ocr_text)
        if hasattr(text_embedding, 'tolist'):
            text_embedding = text_embedding.tolist()
        print(f"Indexing image OCR text with embedding shape: {len(text_embedding) if hasattr(text_embedding, '__len__') else 'unknown'}")
        doc_id = sanitize_key(f"imgocr_{file.filename}")
        index_document({"id": doc_id, "type": "text", "content": ocr_text, "embedding": text_embedding})

    return {"status": "success", "filename": file.filename, "ocr_text": ocr_text}

@app.post("/upload/url")
async def upload_url(urls: List[str] = Form(...)):
    processed = []
    for url in urls:
        print("Processing URL:", url)
        text, image_urls = extract_from_url(url)
        # Text preprocessing
        text_chunks = split_text(text)
        for i, chunk in enumerate(text_chunks):
            print(f"Web chunk {i+1}: {chunk[:100]}...")
            embedding = get_text_embedding(chunk)
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            doc_id = sanitize_key(f"web_{url}_{i+1}")
            print(f"Indexing web chunk {i+1} with embedding shape: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
            index_document({"id": doc_id, "type": "text", "content": chunk, "embedding": embedding})
        # Download and preprocess images
        for idx, img_url in enumerate(image_urls):
            try:
                response = requests.get(img_url, stream=True)
                if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                    filename = os.path.basename(urlparse(img_url).path) or f'web_image_{idx}.jpg'
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    with open(file_path, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                    meta = extract_image_and_ocr(file_path)
                    meta['image_url'] = f"/static/{filename}"
                    print(f"Web image metadata: {meta}")
                    embedding = get_image_embedding(file_path)
                    if hasattr(embedding, 'tolist'):
                        embedding = embedding.tolist()
                    print(f"Indexing web image {filename} with embedding shape: {len(embedding) if hasattr(embedding, '__len__') else 'unknown'}")
                    doc_id = sanitize_key(f"webimg_{filename}_{idx+1}")
                    index_document({"id": doc_id, "type": "image", "meta": meta, "embedding": embedding})
                    print(f"Downloaded and processed image from URL: {img_url} -> {filename}")
            except Exception as e:
                print(f"Error downloading {img_url}: {e}")
        processed.append(url)
    return {"status": "URLs processed", "urls": processed}

# @app.post("/search/text")
# async def search_text(query: str = Form(...)):
#     embedding = get_text_embedding(query)
#     if hasattr(embedding, 'tolist'):
#         embedding = embedding.tolist()
#     results = search_text_embedding(embedding)
#     return {"results": results}


# Helper: cosine similarity
import numpy as np
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_vector_db(query_embedding, top_k=5, mode='text'):
    """
    Search indexed documents/images for top_k most similar items to query_embedding.
    Returns list of dicts: {'text': ..., 'image_url': ..., 'score': ...}
    """
    
    client = get_search_client()
    # Use vector search if available, else fallback to text search
    results = []
    # Example: search all docs, compute similarity
    docs = client.search("*")  # Get all docs (replace with vector search if available)
    for doc in docs:
        embedding = doc.get('embedding')
        if embedding:
            # Skip if embedding shapes do not match
            try:
                if len(query_embedding) != len(embedding):
                    continue
                score = cosine_similarity(query_embedding, embedding)
            except Exception as e:
                print(f"Embedding shape error: {e}")
                continue
            meta = doc.get('meta') or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            result = {
                'text': doc.get('content') or meta.get('ocr_text', ''),
                'image_url': meta.get('image_url', None),
                'meta': meta,
                'score': score
            }
            results.append(result)
    # Sort by score descending
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:top_k]
    return results


@app.post("/search/text")
async def search_text(query: str = Form(...)):
    embedding = get_text_embedding(query)
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    results = search_vector_db(embedding, top_k=5, mode='text')
    return {"results": results}

# @app.post("/search/image")
# async def search_image(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as f:
#         f.write(await file.read())
#     embedding = get_image_embedding(file_path)
#     if hasattr(embedding, 'tolist'):
#         embedding = embedding.tolist()
#     results = search_image_embedding(embedding)
#     return {"results": results}

def save_upload(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    img_path = save_upload(file)
    embedding = get_image_embedding(img_path)
    if hasattr(embedding, 'tolist'):
        embedding = embedding.tolist()
    results = search_vector_db(embedding, top_k=5, mode='image')
    return {"results": results}


#  Azure AI Search 
AZURE_SEARCH_ENDPOINT = "https://multimodalsearch2.search.windows.net"
AZURE_SEARCH_INDEX = "index-2"
AZURE_SEARCH_API_KEY = "U3UCrPsrhEi7zQKeHIHLpq6jQLIEToR4dFUSgoFqcOAzSeDH7um0"

def get_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

def azure_search_upload_documents(documents):
    client = get_search_client()
    result = client.upload_documents(documents)
    print("Azure Search indexing result:", result)
    return result

def azure_search_query(query_text, top=5):
    client = get_search_client()
    results = client.search(query_text, top=top)
    return [doc for doc in results]



