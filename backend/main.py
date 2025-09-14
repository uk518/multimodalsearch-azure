from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List


import os
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/pdf")
async def upload_pdf(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"Saved PDF: {file.filename}")
        saved_files.append(file.filename)
    return {"status": "PDFs uploaded", "filenames": saved_files}

@app.post("/upload/image")
async def upload_image(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"Saved image: {file.filename}")
        saved_files.append(file.filename)
    return {"status": "Images uploaded", "filenames": saved_files}


import requests
from urllib.parse import urlparse

@app.post("/upload/url")
async def upload_url(urls: List[str] = Form(...)):
    saved_images = []
    print("Received URLs:", urls)
    for url in urls:
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                filename = os.path.basename(urlparse(url).path) or 'downloaded_image.jpg'
                file_path = os.path.join(UPLOAD_DIR, filename)
                with open(file_path, 'wb') as out_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        out_file.write(chunk)
                print(f"Downloaded image from URL: {url} -> {filename}")
                saved_images.append(filename)
            else:
                print(f"URL is not an image or failed to download: {url}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    return {"status": "URLs ingested", "images": saved_images, "urls": urls}

@app.post("/search/text")
async def search_text(query: str = Form(...)):
    # TODO: Search embeddings in Azure AI Search
    return {"results": []}

@app.post("/search/image")
async def search_image(file: UploadFile = File(...)):
    # TODO: Search image embeddings in Azure AI Search
    return {"results": []}
