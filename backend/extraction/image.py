from PIL import Image
import pytesseract
import os

def extract_image_and_ocr(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    meta = {
        'format': img.format,
        'size': os.path.getsize(image_path),
        'dimensions': img.size
    }
    return text, meta
