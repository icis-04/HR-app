from anthropic import Anthropic
import voyageai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()
mbd_client = voyageai.Client()

EMBED_MODEL = "voyage-3-large"     # Voyage's flagship model, closest equivalent to text-embedding-3-large
EMBED_DIMENSION = 2048
EMBED_DTYPE = "float"   

# --- Chunking config ---
CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 100      # overlap between chunks, helps preserve context across boundaries

splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts= [d.text for d in docs if getattr(d, "text", None)]
    chunks= []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of document chunks with Voyage AI for storage/retrieval."""
    result = mbd_client.embed(
        texts=chunks,
        model=EMBED_MODEL,
        input_type="document",
        output_dimension=EMBED_DIMENSION,
        output_dtype=EMBED_DTYPE,
    )
    return result.embeddings

def embed_query(query: str) -> list[float]:
    """Embed a single user query with Voyage AI for similarity search."""
    result = mbd_client.embed(
        texts=[query],
        model=EMBED_MODEL,
        input_type="query",
        output_dimension=EMBED_DIMENSION,
        output_dtype=EMBED_DTYPE,
    )
    return result.embeddings[0]

