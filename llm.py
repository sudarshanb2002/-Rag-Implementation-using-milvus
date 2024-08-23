import json
import numpy as np
import torch
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType, exceptions
from transformers import AutoTokenizer, AutoModel

# Initialize Hugging Face model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    """Generate embedding for a given text using BERT."""
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return  embeddings.astype(np.float32)
    # return embeddings_vector

def connect_to_milvus():
    """Connect to the Milvus database."""
    try:
        print("Trying to Connect............")
        connections.connect(alias="default", host="localhost", port="19530")
        print("Connected to Milvus successfully.")
    except exceptions.ConnectError as e:
        print(f"Failed to connect to Milvus: {e}")
        raise e

def truncate_text(text, max_length):
    """Truncate text to the maximum allowed length."""
    return text[:max_length]

def create_index(collection):
    """Create an index on the embedding field."""
    index_params = {
        "metric_type": "L2",  # Use Euclidean distance for similarity search
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created successfully.")

def create_index_if_needed(collection):
    """Check if the index exists; create it if not."""
    if not collection.has_index():
        create_index(collection)
def check_index(collection):
    """Check if the index is properly created on the collection."""
    if collection.has_index():
        print("Index exists on the collection.")
        # Optional: Print the index parameters (this may need to be adjusted based on your SDK version)
        index_info = collection.index()
        print(f"Index information: {index_info}")
    else:
        print("No index found on the collection.")

def insert_to_milvus(content):
    """Insert content into the Milvus collection."""
    try:
        connect_to_milvus()
       
        collection_name = 'wikipedia_data'

        # Check if the collection exists
        if utility.has_collection(collection_name):
            # Drop the existing collection
            collection = Collection(name=collection_name)
            collection.drop()
            print(f"Dropped existing collection: {collection_name}")

        # Define the schema for the collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="paragraphs", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="lists", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
        ]
        schema = CollectionSchema(fields, description="Wikipedia Data Collection")
        
        # Create the collection
        collection = Collection(name=collection_name, schema=schema)
        create_index(collection)  # Assuming this is a function to create an index
        
        print(f"Created new collection: {collection_name}")

        # Prepare and insert data
        title = content["title"]
        headings = ", ".join(content["headings"])
        paragraphs = " ".join(content["paragraphs"])
        lists = " ".join(content["lists"])
        description = content.get("description", "")

        title = truncate_text(title, 500)
        headings = truncate_text(headings, 500)
        paragraphs = truncate_text(paragraphs, 4500)
        lists = truncate_text(lists, 4500)
        description = truncate_text(description, 4500)

        embedding_text = f"{title} {headings} {paragraphs} {lists}".strip()
        embedding = get_embedding(embedding_text)
        embedding = embedding.astype(np.float32).tolist()
        data_to_insert = [
            [title],
            [headings],
            [paragraphs],
            [lists],
            [description],
            [embedding]
        ]

        # Insert data into the collection
        collection.insert(data_to_insert)
        print("Data inserted successfully.")
    except exceptions.CollectionNotExistException as e:
        print(f"Collection does not exist: {e}")
        raise e
    except exceptions.ParamError as e:
        print(f"Invalid data format: {e}")
        raise e
    except Exception as e:
        print(f"Failed to insert data into Milvus: {e}")
        raise e

def perform_similarity_search(query_text, top_k=5):
    """Search for similar content in Milvus based on the query text."""
    # Connect to Milvus
    connect_to_milvus()
    
    
    # Define the collection name
    collection_name = 'wikipedia_data'
    
    # Prepare the collection object
    collection = Collection(name=collection_name)
    # Ensure the collection is loaded
     # Load the collection
    try:
        collection.load()
        print(f"Collection {collection_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading collection {collection_name}: {e}")
        return []
    # Generate the embedding for the query text
    query_embedding = get_embedding(query_text)


    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}


    top_k = 5

    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
    data=[query_embedding.tolist()],
    anns_field="embedding",
    param=search_params,
    limit=10,
    # output_fields=["title", "headings", "paragraphs", "description"]
    output_fields=["paragraphs"]
)
    

    distances = []
    for result in results:
        for item in result:
            distance = item.distance
            distances.append(distance)

    # Format as a dictionary with a single key 'distance'
    result_json = {"Similarity Search Distance": distances}
    return result_json

