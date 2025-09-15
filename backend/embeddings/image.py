#  CLIP embedding stub
#  OpenAI CLIP or transformers


def get_image_embedding(image_path):
    """
    Generate image embeddings using CLIP (OpenAI or HuggingFace transformers).
    Returns a list of floats (embedding vector).
    """
    try:
        from PIL import Image
        import torch
        from transformers import CLIPProcessor, CLIPModel
        # Load model and processor (cache for performance)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        embedding = image_features.squeeze().cpu().numpy().tolist()
        return embedding
    except Exception as e:
        print(f"CLIP embedding error: {e}")
        return [0.0] * 512
