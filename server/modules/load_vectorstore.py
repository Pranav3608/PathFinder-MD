import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "meditron"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok = True)

# intializing pinecone instance
pc = Pinecone(api_key = PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes = [i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name = PINECONE_INDEX_NAME,
        dimension = 768,
        metric = "dotproduct",
        spec = spec
    )

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(PINECONE_INDEX_NAME)


# load, split, embed and upsert pdf docs content
def load_vectorstore(uploaded_files):
    embed_model = HuggingFaceEmbeddings(model_name="NeuML/pubmedbert-base-embeddings")
    file_paths = []

    # 1. Uploading
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # 2. Splitting, Embedding, and Upserting
    for file_path in file_paths: # NOTE: Using file_paths here
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]

        # CORRECTED: Use len(chunks) to get the size for range()
        ids = [f"{Path(file_path).stem} - {i}" for i in range(len(chunks))]

        # Embedding is done in a single call for efficiency
        print(f"Embedding {len(chunks)} chunks...")
        embedding = embed_model.embed_documents(texts)
        
        # 3. Upserting (Recommended: Batching for stability and progress tracking)
        BATCH_SIZE = 100 # Define your batch size
        
        # Combine vectors, ids, and metadata into iterable data
        data_to_upsert = zip(ids, embedding, metadata)
        
        print(f"Upserting embeddings in batches of {BATCH_SIZE}...")
        
        # Use tqdm to show progress through the batches
        for i in tqdm(range(0, len(embedding), BATCH_SIZE), desc="Upserting to Pinecone"):
            i_end = min(i + BATCH_SIZE, len(embedding))
            batch = [next(data_to_upsert) for _ in range(BATCH_SIZE) if i + _ < len(embedding)]

            # The upsert call expects (id, vector, metadata)
            index.upsert(vectors=batch)
        
        print(f"Upload complete for {Path(file_path).name}")

    return {"message": f"Successfully uploaded {len(uploaded_files)} file(s)."}