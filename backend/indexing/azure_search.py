import numpy as np
import matplotlib.pyplot as plt

def plot_similarity_heatmap(query_embedding, result_embeddings, result_ids=None, save_path=None):
    """
    Plots a heatmap of cosine similarity between the query embedding and result embeddings.
    query_embedding: list or np.array
    result_embeddings: list of lists or np.array
    result_ids: optional list of document ids/names for labeling
    save_path: optional path to save the heatmap image
    """
    query_vec = np.array(query_embedding)
    result_vecs = np.array(result_embeddings)
    # Normalize vectors
    query_vec = query_vec / np.linalg.norm(query_vec)
    result_vecs = result_vecs / np.linalg.norm(result_vecs, axis=1, keepdims=True)
    # Compute cosine similarity
    similarities = np.dot(result_vecs, query_vec)
    # Reshape for heatmap (1 row, N columns)
    heatmap = similarities.reshape(1, -1)
    plt.figure(figsize=(max(6, len(similarities)), 2))
    plt.imshow(heatmap, cmap='viridis', aspect='auto')
    plt.colorbar(label='Cosine Similarity')
    if result_ids:
        plt.xticks(ticks=np.arange(len(result_ids)), labels=result_ids, rotation=45, ha='right')
    else:
        plt.xticks(ticks=np.arange(len(similarities)), labels=[str(i+1) for i in range(len(similarities))], rotation=45, ha='right')
    plt.yticks([])
    plt.title('Similarity Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show()

# Azure AI Search integration
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

# Azure credentials
AZURE_SEARCH_ENDPOINT = "https://multimodalsearch2.search.windows.net"
AZURE_SEARCH_INDEX = "index-2"
AZURE_SEARCH_API_KEY = "U3UCrPsrhEi7zQKeHIHLpq6jQLIEToR4dFUSgoFqcOAzSeDH7um0"

def get_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

def index_document(doc):
    """
    Index a document in Azure AI Search.
    doc: dict with keys like id, content, embedding, type
    """
    client = get_search_client()
    result = client.upload_documents([doc])
    print("Azure Search indexing result:", result)
    return result

def search_text_embedding(query_text, top=5):
    """
    Search for documents using text query.
    query_text: str
    """
    client = get_search_client()
    results = client.search(query_text, top=top)
    return [doc for doc in results]

def search_image_embedding(embedding, top=5):
    """
    Search for documents using image embedding (vector search).
    embedding: list of floats
    """
    client = get_search_client()
    
    vector_query = {
        "vector": {
            "value": embedding,
            "fields": "embedding",
            "k": top
        }
    }
    results = client.search("*", vector_queries=[vector_query], top=top)
    return [doc for doc in results]
