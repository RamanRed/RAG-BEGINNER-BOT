import os
import shutil
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer

# Define the data path
DataPath = "data/books"
CHROMA_PATH = "./chromadb"
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
# Remove existing ChromaDB storage
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

# Initialize ChromaDB
chromadb_instance = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chromadb_instance.get_or_create_collection(name="myEmbeddings")

# 🔹 Step 1: Load Documents
def LoadDocument():
    loaded = DirectoryLoader(DataPath, glob="**/*.md")
    document = loaded.load()
    return document

# 🔹 Step 2: Split Documents into Chunks
def DataSplitter(document):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(document)  # Fix: Define `chunks`
    print(f"Number of original documents: {len(document)} | Number of chunks: {len(chunks)}")
    return chunks

# 🔹 Step 3: Save Chunks and Embeddings to ChromaDB
def SaveToChroma(chunks):
    # Prepare data for storage
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_ids = [str(i) for i in range(len(chunks))]  # Unique IDs for each chunk
    chunk_embeddings = model.encode(chunk_texts).tolist()  # Generate embeddings

    # Store embeddings in ChromaDB
    collection.add(
        ids=chunk_ids,  # Unique chunk IDs
        embeddings=chunk_embeddings,  # Vector embeddings
        metadatas=[{"text": text} for text in chunk_texts]  # Store raw text
    )

    print(f"✅ Saved {len(chunks)} chunks to ChromaDB at {CHROMA_PATH}.")

# 🔹 Step 4 Add the query search
def QuerySerach(query, topk):
    EmbeddedQuery = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[EmbeddedQuery],
        n_results=topk
    )

    print("Raw Query Output:", results)  # Debugging step

    # 🔹 Extract and display retrieved documents
    retrieved_texts = [meta.get("text", "No text found") for meta in results.get("metadatas", [{}])[0]]
    retrieved_scores = results.get("distances", [[]])[0]

    # 🔹 Print results
    for i, (text, score) in enumerate(zip(retrieved_texts, retrieved_scores)):
        print(f"\n🔹 Result {i+1} (Score: {score:.4f}):\n{text}")

    return retrieved_texts
# 🔹 Step 5 Run the Process
documents = LoadDocument()
chunks = DataSplitter(documents)
SaveToChroma(chunks)
QuerySerach("What is the title of the eBook mentioned, who is its author, and under what license is it distributed?", 3)
