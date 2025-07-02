from weaviate.classes.config import Configure
from weaviate.collections.classes.config import Property, DataType, Configure
from weaviate.client import WeaviateClient
from app.embeddings.embed import embed_text
from app.weaviate.utility import uuid_from_string
import json


# Create a collection in Weaviate if it doesn't exist
def create_chunks_collection(
    client: WeaviateClient, 
    collection_name: str,
    vectorizer_config=Configure.Vectorizer.none(),  # Disable automatic vectorization by Weaviate
):
    if not client.collections.exists(collection_name):
        print("Creating collection ", collection_name)
        client.collections.create(
            name=collection_name,
            properties=[Property(name="data", data_type=DataType.TEXT)],  # Define 'data' property as text
            vectorizer_config=vectorizer_config,  # Use passed vectorizer config
        )


# Batch insert multiple chunks (text pieces) into the Weaviate collection
def batch_insert_chunks(
    client: WeaviateClient, 
    chunks, 
    collection_name: str,
):
    collection = client.collections.get(collection_name)
    object_vectors = embed_text(chunks)  # Compute vector embeddings for all chunks at once

    print("Batch inserting chunks")
    # Use fixed-size batch to efficiently insert data
    with collection.batch.fixed_size(batch_size=64) as batch:
        for obj, vector in zip(chunks, object_vectors):
            # Add each object with its vector and deterministic UUID generated from chunk content
            batch.add_object(
                properties={"data": obj},
                vector=vector,
                uuid=uuid_from_string(json.dumps(obj, sort_keys=True))
            )
            # Stop batch import if too many errors occur
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    # After batch insertion, check if any objects failed
    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")


# Retrieve top k most similar chunks to a query, filtered by similarity threshold
from weaviate.classes.query import MetadataQuery

def get_top_k_chunks(
    client, 
    collection_name: str, 
    query_text: str, 
    k: int = 2,
    similarity_threshold: float = 0.85
):
    print("Getting top", k, "chunks with similarity threshold", similarity_threshold)

    collection = client.collections.get(collection_name)
    near_vector = embed_text(query_text)  # Embed the query text to a vector

    # Query the collection for the closest vectors to the query embedding
    response = collection.query.near_vector(
        near_vector=near_vector,
        limit=k,
        return_metadata=MetadataQuery(distance=True)  # Get the distance metric (cosine distance)
    )

    filtered_results = []
    for obj in response.objects:
        # Convert distance to cosine similarity (cosine_sim = 1 - distance)
        cosine_sim = 1 - obj.metadata.distance
        print("Cosine sim:", cosine_sim)
        if cosine_sim >= similarity_threshold:
            filtered_results.append(obj)  # Only keep results above similarity threshold

    return filtered_results
