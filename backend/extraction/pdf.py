import pdfplumber

def extract_text_and_images(pdf_path):
    text = ""
    images = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            images.extend(page.images)
    return text, images
