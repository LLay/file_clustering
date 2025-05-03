import openai
from openai.types.create_embedding_response import CreateEmbeddingResponse

import numpy as np
import os


openai.api_key = "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"

SYSTEM_PROMPT = """
You are a technical summarizer.
Summarize the document in under 1000 words
Do not add opinions or make assumptions.
"""

USER_PROMPT = """You are an expert document summarizer. Your task is to extract key information from the provided text to help identify similar or duplicate documents.

Instructions for summarizing the document:
- Write the title of the file
- Write the author of the file
- Write the date of the file
- Write the names mentioned in the file
- Write the organizations and other important entities mentioned in the file
- Summarize the whole file in a paragraph
"""

SUMMARIZATION_MODEL = "gpt-4o"
SUMMARIZATION_MAX_TOKENS = 1000

TEXT_EMBEDDING_MODEL = "text-embedding-3-large"

COSINE_SIMILARITY_THRESHOLD = 0.9


def upload_file(file_path: str) -> str:
    """Upload a file to OpenAI and return the file ID."""
    with open(file_path, "rb") as file:
        response = openai.files.create(
            file=file,
            purpose="assistants"
        )
    return response.id

def get_summary(file_id: str) -> str:
    response = openai.chat.completions.create(
        model=SUMMARIZATION_MODEL,
        temperature=0.1,
        max_tokens=SUMMARIZATION_MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    { "type": "file", "file": {"file_id": file_id} },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
    )
    content = response.choices[0].message.content
    assert content is not None
    return content.strip()

def get_document_embedding(file_path: str) -> np.ndarray:
    """Creates document embedding by summarizing the document and embedding the summary."""
    print("Processing file:", file_path)
    # Step 1) Upload the file
    file_id = upload_file(file_path)
    # Step 2) Get the summary
    summary = get_summary(file_id)
    print("Summary:", summary)
    # Step 3) Get the embedding
    embedding_response = openai.embeddings.create(
        input=summary,
        model=TEXT_EMBEDDING_MODEL
    )
    return np.array(embedding_response.data[0].embedding)

def main(directory) -> None:
    file_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".pdf"))
    ]
    embeddings = []
    for path in file_paths:
        embeddings.append(get_document_embedding(path))
    embeddings = np.array(embeddings) 
    # Normalize embedding norm
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings /= norms[:, np.newaxis]
    # Compute pairwise cosine similarity
    similarity_matrix = np.dot(embeddings, embeddings.T)
    print(similarity_matrix)
    # For each file path, print the similarity scores with other file paths
    for i, path in enumerate(file_paths):
        print(f"File: {path}")
        for j, score in enumerate(similarity_matrix[i]):
            if i != j:
                print(f"  Similarity with {file_paths[j]}: {score:.4f}")
        print()
    # Print the document pairs that have a similarity score above the threshold
    print("Document pairs with similarity above threshold:")
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            if similarity_matrix[i][j] > COSINE_SIMILARITY_THRESHOLD:
                print(f"{file_paths[i]} and {file_paths[j]}: {similarity_matrix[i][j]:.4f}")

# Run the pipeline
main("./test_files")
