from datetime import datetime, timedelta
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.logger import logger


def normalize_dates(text):

    today = datetime.today()

    if "tomorrow" in text.lower():
        tomorrow = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        text = re.sub(r"tomorrow", tomorrow, text, flags=re.I)

    if "today" in text.lower():
        today_str = today.strftime("%Y-%m-%d")
        text = re.sub(r"today", today_str, text, flags=re.I)

    return text


def load_and_preprocess(file_path):

    logger.info("PREPROCESS: Opening file")

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    logger.info("PREPROCESS: File loaded")

    # Basic cleaning
    text = text.replace("\n", " ")
    text = " ".join(text.split())

    logger.info("PREPROCESS: Text cleaned")


    # -------------------------
    # Chunking
    # -------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    documents = splitter.split_text(text)

    logger.info(f"PREPROCESS: Total chunks created: {len(documents)}")

    return documents