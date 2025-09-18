from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

def get_search_client():
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
    )

def inspect_index():
    client = get_search_client()
    print("Querying all indexed documents...")
    results = client.search("*")
    for doc in results:
        print("ID:", doc.get("id"))
        print("Type:", doc.get("type"))
        print("Meta:", doc.get("meta"))
        print("Embedding shape:", len(doc.get("embedding", [])))
        print("Content:", doc.get("content"))
        print("---")

if __name__ == "__main__":
    inspect_index()