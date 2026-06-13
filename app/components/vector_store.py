from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embedding_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os

logger = get_logger(__name__)
def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading EXisting vectorstore.....")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embedding_model,
                allow_dangerous_deserialization = True
            )
        else:
            logger.warning("No vectorstore found.....")    
            
            
    except Exception as e:
        error_message = CustomException("Failed to load vectorstore ",e)
        logger.error(str(error_message))
        raise error_message         
    
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks are found")
        logger.info("Generating your new vector store")
        
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(text_chunks,embedding_model)
        
        logger.info("Saving vectorstore....")
        db.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore save successfully.....")
        return db
    
    except Exception as e:
        error_message = CustomException("Failed to create vectorstore ",e)
        logger.error(str(error_message))
        raise error_message