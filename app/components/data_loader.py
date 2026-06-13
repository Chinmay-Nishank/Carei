import os 
from app.components.load_pdf import load_pdf_files,create_text_chunks
# no embedding model load required (import)
from app.components.vector_store import save_vector_store
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def process_store_pdf():
    try:
        logger.info("Making the Vectorstore.......")
        documents = load_pdf_files()
        text_chunks = create_text_chunks(documents)
        save_vector_store(text_chunks)
        logger.info("Vectorstore created successfully.....")

    except Exception as e:
        error_message = CustomException("Failed to create new vector",e)
        logger.error(str(error_message))
        
if __name__ == "__main__":
    process_store_pdf()
            
        
    