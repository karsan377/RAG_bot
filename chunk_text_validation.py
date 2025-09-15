from langchain_community.vectorstores import Chroma
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

source_ids = [
    "data/book.pdf:37:3",
    "data/book.pdf:18:0",
    "data/book.pdf:106:1",
    "data/book.pdf:27:2",
    "data/book.pdf:95:0"
]

# Load Chroma database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

# Fetch documents and metadata
all_docs = db.get(include=["documents", "metadatas"])

# Create a map from chunk ID (stored in metadata) to text
docs_by_id = {meta["id"]: doc for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])}

# Print the chunks
for src_id in source_ids:
    text = docs_by_id.get(src_id)
    if text:
        print(f"ID: {src_id}\nText:\n{text}\n{'-'*60}")
    else:
        print(f"ID: {src_id} not found in database")
