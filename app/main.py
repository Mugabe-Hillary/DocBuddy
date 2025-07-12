import streamlit as st
import os
import sys
from dotenv import load_dotenv
import logging

# Load environment variables from the root directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline
from utils import get_vector_store
from config import Config

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE, 
    page_icon=Config.PAGE_ICON, 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title(f"📚 {Config.PAGE_TITLE}")
st.markdown("*Wanna talk about your document*")

# Check for API key
if not Config.GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it to use the application.")
    st.stop()

@st.cache_resource
def initialize_pipeline():
    """Initializes and caches the RAG pipeline with a persistent vector store."""
    logger.info("Initializing RAG pipeline...")
    
    # The vector store is now persistent and doesn't rely on a default document
    vectorstore = get_vector_store()
    
    if vectorstore is not None:
        logger.info("Successfully loaded vector store.")
        return RAGPipeline(vectorstore)
    
    logger.error("Failed to initialize vector store.")
    return None

# Initialize the pipeline
try:
    st.session_state.rag_pipeline = initialize_pipeline()
except Exception as e:
    logger.error(f"Error during pipeline initialization: {e}")
    st.session_state.rag_pipeline = None

if not st.session_state.rag_pipeline:
    st.error("❌ Failed to initialize the application.")
    st.stop()
else:
    st.sidebar.success("✅ Application initialized successfully!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    uploaded_file = st.file_uploader(
        "Upload a text file to add to the knowledge base",
        type=["txt", "md"],
        help="Your document will be processed and added to the persistent vector store."
    )
    
    if uploaded_file is not None:
        if st.button("📤 Process Document", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Save uploaded file temporarily
                    temp_path = f"/tmp/{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    from utils import load_document, chunk_text
                    
                    text = load_document(temp_path)
                    if text:
                        chunks = chunk_text(text)
                        
                        # Get the vector store and add new texts
                        vectorstore = st.session_state.rag_pipeline.vectorstore
                        vectorstore.add_texts(chunks)
                        
                        st.success(f"✅ Added {uploaded_file.name} to the knowledge base!")
                    else:
                        st.error("❌ Failed to process document.")
                        
                    os.remove(temp_path)
                    
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
                    st.error("An error occurred while processing the document. Please check the logs.")
    
    st.divider()
    
    st.header("⚙️ Settings")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    temperature = st.slider("🌡️ Temperature", Config.MIN_TEMPERATURE, Config.MAX_TEMPERATURE, Config.DEFAULT_TEMPERATURE, 0.1)
    max_tokens = st.slider("📏 Max Tokens", Config.MIN_MAX_TOKENS, Config.MAX_MAX_TOKENS, Config.DEFAULT_MAX_TOKENS, 100)
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown(f"""
    **{Config.PAGE_TITLE}** lets you get to know more about documents using AI.
    
    **Features:**
    - 🔍 Semantic search
    - 🤖 AI-powered answers
    - 📤 Upload your own files
    """)

    st.divider()
    st.markdown("*Developed by AMH© 2025*")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question, I don't mind😉"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_pipeline.query(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


