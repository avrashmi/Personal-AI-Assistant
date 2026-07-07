# Not in use
import logging

def setup_logger():
    """
    Initializes and configures the application-wide logger.

    Features:
    - Logs messages to both console and a file (app.log)
    - Includes timestamp, log level, and message
    - Helps debug RAG pipeline (retrieval, LLM, evaluation)

    Why important:
    - Tracks internal system behavior
    - Essential for debugging production issues
    - Provides traceability for each user request
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("agentic-rag")


logger = setup_logger()


'''import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger("agentic-rag")

logger = setup_logger()'''