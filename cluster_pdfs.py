import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse

import pdfplumber
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
import os

# Set your OpenAI API key
openai.api_key = "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"


### ---- 1. Extract Text from PDFs and DOCX ---- ###
def extract_text_from_pdf(pdf_path, max_pages=5):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(min(max_pages, len(pdf.pages))):
            text.append(pdf.pages[page_num].extract_text() or "")  # Handle empty pages
    return "\n".join(text).strip()


def extract_text(file_path, max_pages=5):
    print(f"Extracting text from: {file_path}")
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path, max_pages)
    return ""


### ---- 2. Generate OpenAI Embeddings ---- ###
def get_openai_embedding(text):
    print(f"Generating embedding for: {text[:50]}...")
    if not text.strip():
        return np.zeros(1536)  # OpenAI embeddings are 1536-dimensional
    response: CreateEmbeddingResponse = openai.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)


def generate_embeddings(documents):
    return np.array([get_openai_embedding(doc) for doc in documents])


### ---- 3. Compute Similarity Matrix ---- ###
def compute_similarity_matrix(embeddings):
    print("Computing similarity matrix...")
    return cosine_similarity(embeddings)


### ---- 4. Apply Hierarchical Clustering ---- ###
def hierarchical_clustering(similarity_matrix, threshold=0.3):
    print(f"Applying hierarchical clustering with threshold: {threshold}")
    # Convert similarity to distance (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    linkage_matrix = linkage(distance_matrix, method="ward")
    clusters = fcluster(linkage_matrix, threshold, criterion="distance")
    return clusters, linkage_matrix


### ---- 5. Plot Dendrogram (Optional) ---- ###
def plot_dendrogram(linkage_matrix, file_names):
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=file_names, leaf_rotation=90, leaf_font_size=10)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Documents")
    plt.ylabel("Distance")
    plt.show()


### ---- MAIN PIPELINE ---- ###
def main(directory, max_pages=5, clustering_threshold=0.3):
    # Load documents
    file_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".pdf"))
    ]
    file_names = [os.path.basename(f) for f in file_paths]

    # Extract text
    documents = [extract_text(f, max_pages) for f in file_paths]
    print(f"Extracted text from {len(documents)} documents.")

    # Generate embeddings
    embeddings = generate_embeddings(documents)
    print("Generated embeddings.")

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Perform hierarchical clustering
    clusters, linkage_matrix = hierarchical_clustering(
        similarity_matrix, threshold=clustering_threshold
    )

    # Output clusters
    clustered_docs = {}
    for i, cluster_id in enumerate(clusters):
        clustered_docs.setdefault(cluster_id, []).append(file_names[i])

    print("\nDocument Clusters:")
    for cluster_id, docs in clustered_docs.items():
        print(f"Cluster {cluster_id}: {docs}")

    # Optional: Plot Dendrogram
    plot_dendrogram(linkage_matrix, file_names)


# Run the pipeline
main("./test_files", max_pages=5, clustering_threshold=0.3)
