import os
import requests
import asyncio
import aiohttp
import json
from typing import List
from requests.exceptions import ConnectionError, Timeout, RequestException
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

import logging
from time import sleep

# Set up logging for better debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set a default User-Agent if not in environment variables
USER_AGENT = os.getenv("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36")
HEADERS = {'User-Agent': USER_AGENT}


# Function to check if a document exists in the vectorstore
def document_exists(vectorstore, doc, embedding_function):
    query_text = doc.page_content[:500]  # Using the first 500 characters
    query_embedding = embedding_function.embed_query(query_text)
    results = vectorstore.similarity_search_by_vector(query_embedding, k=1)
    if results:
        for result in results:
            if result.metadata == doc.metadata:
                return True
    return False


#adding mental states diagram

def process_mental_state_model(
    mental_model_jsonl_path: str,
    persist_directory: str,
    embedding_model,
    collection_name: str = "mental_state_model"
):
    """
    Ingest the serialized PlantUML mental state model (transitions + state profiles)
    into a Chroma collection.

    mental_model_jsonl_path = path to a .jsonl file where each line is one JSON object:
      - type: "transition" or "state_profile"
      - fields like from_state, to_state, condition_or_intervention, etc.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Try to load existing vectorstore (if it exists). If not, we'll create it later.
    if os.path.exists(persist_directory):
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        logger.info(f"ℹ️ Loaded existing mental state collection from {persist_directory}")
    else:
        vectorstore = None

    docs_raw: List[Document] = []

    with open(mental_model_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if item.get("type") == "transition":
                text_block = (
                    f"STATE TRANSITION:\n"
                    f"From {item.get('from_state')} to {item.get('to_state')}.\n"
                    f"Condition or intervention: {item.get('condition_or_intervention')}.\n"
                    f"Direction: {item.get('direction')}.\n"
                    f"Notes: {item.get('notes')}"
                )
                metadata = {
                    "source": item.get("source", "MentalStateModel_v1"),
                    "type": "transition",
                    "from_state": item.get("from_state"),
                    "to_state": item.get("to_state"),
                    "direction": item.get("direction"),
                }

            elif item.get("type") == "state_profile":
                text_block = (
                    f"STATE PROFILE: {item.get('state')}.\n"
                    f"Entry signals: {', '.join(item.get('entry_signals', []))}.\n"
                    f"Ongoing pattern: {', '.join(item.get('ongoing_pattern', []))}.\n"
                    f"Exit / recovery pattern: {', '.join(item.get('exit_pattern', []))}.\n"
                    f"Summary: {item.get('summary')}"
                )
                metadata = {
                    "source": item.get("source", "MentalStateModel_v1"),
                    "type": "state_profile",
                    "state": item.get("state"),
                }

            else:
                # skip unexpected rows
                continue

            docs_raw.append(Document(page_content=text_block, metadata=metadata))

    logger.info(f"✅ Loaded {len(docs_raw)} PlantUML-derived mental state docs from {mental_model_jsonl_path}")

    # Split to chunks like the rest of your pipeline
    split_docs = splitter.split_documents(docs_raw)

    # Deduplicate vs existing vectorstore
    if vectorstore:
        new_docs = []
        for doc in split_docs:
            if not document_exists(vectorstore, doc, embedding_model):
                new_docs.append(doc)
        if not new_docs:
            logger.info("✅ No new mental state model docs to add.")
            return
    else:
        new_docs = split_docs

    logger.info(f"🧠 Adding {len(new_docs)} mental state model chunks...")

    # Store/persist
    for attempt in range(3):
        try:
            if vectorstore:
                vectorstore.add_documents(new_docs)
                logger.info(f"✅ {len(new_docs)} state-model docs added.")
            else:
                vectorstore = Chroma.from_documents(
                    documents=new_docs,
                    embedding=embedding_model,
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
                logger.info(f"✅ Created new mental_state_model collection with {len(new_docs)} docs.")
            vectorstore.persist()
            break
        except Exception as e:
            logger.warning(f"⚠️ Failed to store mental state model docs (attempt {attempt+1}): {str(e)}")
            if attempt == 2:
                logger.error("❌ Giving up after 3 attempts.")



# Main function to run mental state model processing
if __name__ == "__main__":
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    print("\n🧠 Building Mental State & Wellbeing Knowledge Base...")

    # Path to JSONL with serialized PlantUML transitions/state profiles
    mental_model_jsonl_path = (
        "mental_state_model.jsonl"
    )

    persist_directory_mental = "mental_state_chroma_db"

    process_mental_state_model(
        mental_model_jsonl_path=mental_model_jsonl_path,
        persist_directory=persist_directory_mental,
        embedding_model=embedding_model,
        collection_name="mental_state_model",
    )

    print("\n✅ Mental state knowledge base successfully created!")
