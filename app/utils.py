import os
from typing import List
import streamlit as st
import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb

from config import Config

# Configure logger
logger = logging.getLogger(__name__)


def load_document(file_path: str) -> str:
    """Loads a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        st.error(f"File not found: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Splits text into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    return chunks


def create_embeddings():
    """Creates text embeddings using Google's service."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY not found in environment variables")
        return None
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        google_api_key=api_key
    )
    return embeddings


def get_vector_store(persist_directory: str = Config.VECTOR_STORE_PERSIST_DIR, collection_name: str = "docbuddy_store"):
    """Loads an existing vector store or creates a new one."""
    embeddings = create_embeddings()
    if not embeddings:
        return None
    
    # Use PersistentClient for robust persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    logger.info(f"Loaded vector store from '{persist_directory}' with collection '{collection_name}'.")
    return vectorstore


def get_retriever(vectorstore, k: int = 5):
    """Creates a retriever from a vector store."""
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={"k": k})
    return None


def format_docs(docs):
    """Formats documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)
