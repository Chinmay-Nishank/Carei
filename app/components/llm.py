from langchain_groq import ChatGroq
from app.config.config import GROQ_API_KEY
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def load_llm(model_name : str="llama-3.3-70b-versatile",groq_api_key:str= GROQ_API_KEY):
    try:
        
        
        logger.info("Loading Llm from Groq....")
    
        llm = ChatGroq(
            groq_api_key = groq_api_key,
            model_name = model_name,
            temperature=0.3,
            max_tokens=2048
        )
        logger.info("Llm loaded Successfully from groq....")
        return llm
    except Exception as e:
        error_message = CustomException("Failed to load embedding model ",e)
        logger.error(str(error_message))
        return None
