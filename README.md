Technical Write up

---
Creating Environment:

pip install -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

Frontend code to run:
cd front-end
npm install
npm start

Backend Code to run:
uvicorn map:app --reload

Azure Portal:
Resources:
Azure AI Search
Azure AI Computer Vision


steps:

word chunking-create vectors using embedding models-store them in a database or vector store-train a model 0r fine tune a pre trained model-on which can do semantic search- can do similarity cosine search -provide output for the text or NLP task-text query or image query. 

-----------------------------------------------------------------------



###### **What Is Batch Processing in Azure AI Search?**

Batch processing means handling large volumes of data (e.g., documents, files, or database records) all at once, instead of real-time or one-by-one.



**Azure AI Search supports batch integration by using:**

* Indexers to pull data from sources (like Blob Storage or SQL)
* Skillsets to apply AI enrichment (optional)
* Indexes to store searchable data



###### **Azure Blob Storage**

**Azure Blob Storage** is Microsoft’s object storage solution for the cloud. "Blob" stands for Binary Large OBject, and it’s designed to store unstructured data like:

* Images
* Videos
* Documents
* Backups
* Logs
* Big data files (e.g., CSV, JSON, Parquet)



**Azure AI Search is a search-as-a-service that uses AI to:**

⦁  Index and search large volumes of data

⦁  Extract information using cognitive skills (e.g., OCR, entity recognition, sentiment)

⦁  Enable semantic search (understanding intent, not just keywords)





**How It Integrates with Data**

**Data Ingestion**

Azure AI Search connects to your data using:

* Indexers: Built-in connectors that crawl data sources.
* APIs / SDKs: Push data programmatically.
* Azure Data Factory or Azure Synapse Pipelines: For large-scale orchestration.



**Indexing Process**

**Here’s how the flow works:**

1. Connect to Data Source-Use an indexer or push data manually.
2. Create Index-Define what fields should be searchable, filterable, facetable, etc.
3. Apply Cognitive Skills (Optional)-Attach a skillset (e.g., OCR, key phrase extraction, language detection).
4. Enrich and Index Data-Data is enriched and stored in a searchable index.
5. Search via API or UI-Use REST API, SDKs, or integrate with apps like web portals or chatbots.



###### **Integration Example: Azure Blob + Azure AI Search**

Suppose you have thousands of PDFs in Blob Storage.

1. Blob Storage – stores raw PDFs
2. Indexer – scans each file
3. Skillset – extracts text, applies OCR, detects language
4. Index – stores enriched, searchable content
5. Client App – users search for content by keyword or semantic query



###### **Semantic \& AI Features**

**Azure AI Search adds intelligence beyond keyword matching:**

**Feature	                    Description-**

Semantic Search	            Understands meaning, ranks results by intent

Cognitive Skills	    Prebuilt AI (OCR, translation, entity extraction)

Custom Skills	            Plug in your own ML model via an Azure Function

Vector Search	            Enables search over embeddings (e.g., from OpenAI, BERT) for similarity search

Multi-language Support	    Over 70 languages supported

Facets \& Filters	    Filter by categories like date,type,etc.





###### **What Happens With Document Intelligence**

###### **When you use Azure Document Intelligence:**

You upload a PDF or image

It automatically:

* Extracts all text (OCR)
* Understands layout: tables, columns, form fields
* Identifies key-value pairs (e.g., Invoice Date: 2023-09-10)
* Recognizes document types (invoices, receipts, etc.)
* You get back a structured JSON response, ready to store or index



###### &nbsp;**CLIP (Contrastive Language–Image Pretraining) — from OpenAI**

If you're referring to AI / Machine Learning, especially from OpenAI:

🔹 What is CLIP?

CLIP is a vision-language model developed by OpenAI that can:

Understand images and text together

Match images with descriptive text (and vice versa)

* Perform tasks like:
* Zero-shot classification
* Visual search
* Text-to-image or image-to-text embedding



datasets:

https://github.com/clovaai/cord

https://docvqa.org/dataset/

https://www.cs.cmu.edu/~aharley/rvl-cdip/

https://guillaumejaume.github.io/FUNSD/



###### **Why You Need to Preprocess Text-Image Documents**

A text-image document (e.g., a scanned PDF, form, or document with both text and images) is not directly searchable or queryable unless its contents are extracted and structured. Preprocessing ensures that the information can be indexed, searched, and queried properly.



###### **What About Image Queries?**

If you want image-based queries (e.g., find all receipts with a signature), you may:

* Preprocess images with custom image classifiers or vision models
* Extract visual features
* Tag or cluster images
* Store image embeddings in Azure Search for image similarity search



###### **Text Query Over Text-Image Documents**

Use Case: You want to allow users to query "What is the due date on the invoice?" over uploaded invoices.

* OCR detects text from invoice PDF image.
* Form Recognizer extracts "Due Date: 2025-09-30".
* Index structured content and/or embeddings into Azure AI Search.
* Text query ("due date") gets matched to semantic vector or key-value pair.




Links Supporting for learning:
https://medium.com/@tenyks_blogger/multi-modal-image-search-with-embeddings-vector-dbs-cee61c70a88a







