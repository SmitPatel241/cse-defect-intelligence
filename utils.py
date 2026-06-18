import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY: str = os.environ["GOOGLE_API_KEY"]
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]

PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX", "cse-defect-duplicates")
PINECONE_QUERIES_INDEX_NAME: str = os.environ.get("PINECONE_QUERIES_INDEX", "queries-data")
PINECONE_CLOUD: str = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.environ.get("PINECONE_REGION", "us-east-1")
EMBEDDING_MODEL: str = "models/gemini-embedding-001"
RERANK_MODEL: str = os.environ.get("RERANK_MODEL", "gemini-2.5-flash-lite")
EMBEDDING_DIMENSION: int = 3072
QUERIES_EMBEDDING_DIMENSION: int = int(os.environ.get("QUERIES_EMBEDDING_DIMENSION", "1024"))
BATCH_SIZE: int = int(os.environ.get("EMBED_BATCH_SIZE", "10"))
EMBED_BATCH_PAUSE_SEC: float = float(os.environ.get("EMBED_BATCH_PAUSE_SEC", "6.0"))
EMBED_MAX_RETRIES_PER_BATCH: int = int(os.environ.get("EMBED_MAX_RETRIES_PER_BATCH", "8"))
EMBED_QUOTA_RETRY_BASE_SEC: float = float(os.environ.get("EMBED_QUOTA_RETRY_BASE_SEC", "15.0"))
TOP_K_RETRIEVAL: int = 10
TOP_K_RERANK: int = 5
TOP_K_QUERY_RETRIEVAL: int = int(os.environ.get("TOP_K_QUERY_RETRIEVAL", "25"))
TOP_K_QUERY_RERANK_POOL: int = int(os.environ.get("TOP_K_QUERY_RERANK_POOL", "12"))
MIN_QUERY_RELEVANCE_SCORE: int = int(os.environ.get("MIN_QUERY_RELEVANCE_SCORE", "45"))


def safe_str(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def build_combined_text(summary: str, description: str, comments: str) -> str:
    return (
        f"Summary: {safe_str(summary)}\n"
        f"Description: {safe_str(description)}\n"
        f"Comments: {safe_str(comments)}"
    )


def normalize_l2(vector: list[float]) -> list[float]:
    """L2-normalize an embedding vector (required for non-3072 gemini-embedding-001 dims)."""
    import math

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def retry(max_attempts: int = 3, initial_delay: float = 2.0, backoff: float = 2.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s). Retrying in %.1fs…",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator


def get_gemini_client():
    import google.generativeai as genai  # noqa: PLC0415
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai


def ensure_pinecone_index(dimension: int = EMBEDDING_DIMENSION):
    from pinecone import Pinecone, ServerlessSpec  # noqa: PLC0415

    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing:
        logger.info(
            "Creating Pinecone index '%s' (dim=%d, %s/%s)…",
            PINECONE_INDEX_NAME,
            dimension,
            PINECONE_CLOUD,
            PINECONE_REGION,
        )
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            logger.info("Waiting for index to become ready…")
            time.sleep(3)
        logger.info("Index created and ready.")
    else:
        logger.info("Using existing Pinecone index '%s'.", PINECONE_INDEX_NAME)

    return pc


def get_pinecone_index(dimension: int = EMBEDDING_DIMENSION):
    pc = ensure_pinecone_index(dimension=dimension)
    return pc.Index(PINECONE_INDEX_NAME)


def get_pinecone_queries_index(dimension: int = EMBEDDING_DIMENSION):
    """Return the Pinecone index used for historical CSE query / Q&A records."""
    from pinecone import Pinecone  # noqa: PLC0415

    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_QUERIES_INDEX_NAME)
