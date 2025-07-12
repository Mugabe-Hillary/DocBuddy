# DocBuddy: Your Personal Document Query System

This project is a Retrieval Augmented Generation (RAG) system that allows users to chat with their own documents. It's built with Streamlit, Langchain, and Docker, and features a persistent vector store for your knowledge base.

## Project Description

The core of this project is a RAG pipeline that takes a user's question and a collection of documents, retrieves relevant parts of the documents, and uses a Large Language Model (LLM) to generate an answer based on that context. This approach helps to ground the LLM's responses in the provided text, improving factual accuracy.

The application is designed to work exclusively with user-uploaded documents, creating a persistent, personal knowledge base that you can query across sessions.

## Application Architecture

The application consists of the following components:

-   **Streamlit Frontend (`main.py`):** A web interface for users to upload documents and ask questions.
-   **RAG Pipeline (`rag_pipeline.py`):**  Handles the logic for the RAG process, including creating the prompt and invoking the LLM.
-   **Utility Functions (`utils.py`):** Contains helper functions for processing documents (chunking, embedding) and managing the persistent vector store.
-   **Persistent Vector Store (`chroma_db/`):** ChromaDB is used to store document embeddings, ensuring that your knowledge base is saved between sessions.
-   **Docker (`Dockerfile`, `docker-compose.yml`):** Containerizes the application for easy deployment and reproducibility.

### Chunking and Embedding

-   **Chunking Strategy:** The `RecursiveCharacterTextSplitter` from Langchain is used to break down documents. This strategy effectively splits text on semantic boundaries (paragraphs, sentences, words), which helps keep related content together in the same chunk.
-   **Embedding Model:** Google's Generative AI embedding model (`models/embedding-001`) is used via `GoogleGenerativeAIEmbeddings` in Langchain. This is a powerful and efficient model for generating high-quality embeddings.
-   **LLM:** The application uses Google's Gemini Flash model (`gemini-1.5-flash`) for generating responses, providing fast and accurate natural language understanding.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

-   Docker and Docker Compose
-   A Google API Key (for Google Generative AI / Gemini)

### Installation

1.  **Clone the repository**
    ```bash
    git clone git@github.com:Mugabe-Hillary/DocBuddy.git
    cd DocBuddy
    ```

2.  **Set up environment variables**
    Create a `.env` file in the root of the project and add your Google API key:
    ```
    GOOGLE_API_KEY="your_google_api_key"
    ```

3.  **Build and run the Docker container**
    ```bash
    docker-compose up --build
    ```

4.  **Access the application**
    Open your web browser and go to `http://localhost:8501`.

## Features

- 📤 **Upload Your Documents**: Build a knowledge base from your own text files.
- 💾 **Persistent Knowledge Base**: Your uploaded documents are automatically saved and available across sessions.
- 🔍 **Semantic Search**: Advanced document search using Google's embedding models.
- 🤖 **AI Chat Interface**: Powered by Google's Gemini Flash model.
- ⚙️ **Customizable Settings**: Adjust response creativity and length.
- 🐳 **Docker Support**: Easy deployment with Docker containers.

## Usage

### Web Interface

1. **Upload Documents**: Use the sidebar to upload one or more text files. These will be added to your persistent knowledge base.
2. **Start a Conversation**: Once your documents are processed, type your question in the chat input to start querying them.
3. **Adjust Settings**: Modify response creativity and length using the sliders in the sidebar.
4. **Clear Chat**: Reset the conversation history using the sidebar button.

### Running Locally (without Docker)

   **Or manually**:
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt

   # Run the Streamlit app
   streamlit run app/main.py
   ```

## Project Structure
```
.
├── .env                  # Stores environment variables (e.g., API keys)
├── .gitignore            # Specifies files to be ignored by Git
├── Dockerfile            # Defines the Docker image for the application
├── README.md             # This file
├── app                   # Main application directory
│   ├── __init__.py
│   ├── config.py         # Application configuration settings
│   ├── main.py           # Streamlit frontend and main application logic
│   ├── rag_pipeline.py   # Core RAG pipeline logic
│   └── utils.py          # Helper functions for vector store and text processing
├── chroma_db/            # Directory for the persistent ChromaDB vector store
├── docker-compose.yml    # Defines the Docker services, networks, and volumes
└── requirements.txt      # Python dependencies

## Technical Details

- **Frameworks**: Streamlit, Langchain
- **LLM**: Google Gemini Flash
- **Embedding Model**: Google `embedding-001`
- **Vector Store**: ChromaDB (Persistent)
- **Containerization**: Docker, Docker Compose

## Future Improvements

-   Support for more document types (PDF, DOCX).
-   User authentication to manage separate knowledge bases.
-   More advanced retrieval strategies.

---
*Developed by AMH© 2025*
