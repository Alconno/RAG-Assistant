import os
import atexit
import weaviate
import weaviate.classes as wvc

_client = None  # Global singleton client instance

def get_weaviate_client():
    global _client

    if _client is None:
        # Read config from environment variables
        cluster_url = os.environ["WEAVIATE_URL"]  
        api_key = os.environ["WEAVIATE_API_KEY_ADMIN"]
        huggingface_key = os.environ["HUGGINGFACE_APIKEY"]

        # Connect to Weaviate Cloud with API key authentication and Hugging Face API key header
        _client = weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=wvc.init.Auth.api_key(api_key),
            headers={"X-HuggingFace-Api-Key": huggingface_key}
        )

        # Ensure client connection is closed when Python exits
        atexit.register(_client.close)

    return _client
