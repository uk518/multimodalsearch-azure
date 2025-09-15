from PIL import Image
import pytesseract
import os


def extract_image_and_ocr(image_path):
    """
    Extract OCR text and metadata (format, size, dimensions) from image.
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    meta = {
        'format': img.format,
        'size': os.path.getsize(image_path),
        'dimensions': {'width': img.width, 'height': img.height}
    }
    return text, meta
