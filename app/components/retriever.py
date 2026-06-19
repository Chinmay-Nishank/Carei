from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """
You are CareBot, an advanced AI Medical Assistant.

Instructions:

1. Use the provided context as the primary source of information.
2. If relevant information exists in the context, prioritize it.
3. If the context does not contain enough information but the question is medical, healthcare, disease, medicine, anatomy, nutrition, fitness, mental health, or treatment related, provide a helpful answer using your medical knowledge.
4. If the question is unrelated to medicine or healthcare, respond with:

"I am a medical assistant and can only answer healthcare and medical-related questions."

5. Never invent research findings, statistics, diagnoses, prescriptions, or medical records.
6. Explain information in simple language.
7. Use headings and bullet points when helpful.
8. Keep answers professional and concise.
9. Return plain text only.
10. Do not generate HTML tags.

Context:
{context}

Question:
{question}

Answer:
"""
def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def create_qa_chain():
    try:
        logger.info("Loading vector store...")

        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not found or empty")

        logger.info("Loading LLM...")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM could not be loaded")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(
                search_kwargs={"k": 10}
            ),
            return_source_documents=False,
            chain_type_kwargs={
                "prompt": set_custom_prompt()
            }
        )

        logger.info("QA Chain created successfully")

        return qa_chain

    except Exception as e:
        error_message = CustomException(
            "Failed to create QA Chain",
            e
        )

        logger.exception(str(error_message))
        raise error_message