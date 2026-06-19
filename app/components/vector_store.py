from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)


def load_vector_store():
    try:
        logger.info("Starting vector store loading...")

        print("STEP 1: Loading embedding model...")
        embedding_model = get_embedding_model()

        print("STEP 2: Embedding model loaded")
        logger.info("Embedding model loaded successfully")

        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(
                f"Vector store directory not found: {DB_FAISS_PATH}"
            )

        logger.info(f"Loading FAISS vector store from {DB_FAISS_PATH}")
        print("STEP 3: Loading FAISS vector store...")

        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info("FAISS vector store loaded successfully")
        print("STEP 4: FAISS vector store loaded successfully")

        return db

    except Exception as e:
        logger.exception("Failed to load vector store")
        raise CustomException(
            "Failed to load vector store",
            e
        )


def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise ValueError("No text chunks found")

        logger.info("Generating new vector store...")

        print("STEP A: Loading embedding model...")
        embedding_model = get_embedding_model()

        print("STEP B: Creating FAISS index...")
        db = FAISS.from_documents(
            text_chunks,
            embedding_model
        )

        os.makedirs(DB_FAISS_PATH, exist_ok=True)

        logger.info(f"Saving vector store to {DB_FAISS_PATH}")
        print("STEP C: Saving FAISS index...")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vector store saved successfully")
        print("STEP D: Vector store saved successfully")

        return db

    except Exception as e:
        logger.exception("Failed to create vector store")
        raise CustomException(
            "Failed to create vector store",
            e
        )