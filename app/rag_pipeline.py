from typing import List
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from utils import get_retriever, format_docs
from config import Config
import os


class RAGPipeline:
    """Handles the RAG pipeline logic."""
    
    def __init__(self, vectorstore, model_name: str = Config.LLM_MODEL):
        """Initializes the RAG pipeline."""
        self.vectorstore = vectorstore
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.retriever = get_retriever(vectorstore)
        self.chain = self._build_chain()
    
    def _initialize_llm(self):
        """Initializes the language model."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=api_key,
            temperature=Config.DEFAULT_TEMPERATURE,
            convert_system_message_to_human=True
        )
    
    def _build_chain(self):
        """Builds the RAG chain."""
        template = """You are a helpful assistant that answers questions based on the provided context from documents. 

Context from documents:
{context}

Question: {question}

Instructions:
- Answer the question based primarily on the provided context
- If the context doesn't contain enough information, say so politely
- Be accurate and cite specific details from the context when possible
- If the question is about Sherlock Holmes stories, provide engaging and detailed answers
- Keep your responses informative but concise

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def query(self, question: str, temperature: float = Config.DEFAULT_TEMPERATURE, max_tokens: int = Config.DEFAULT_MAX_TOKENS) -> str:
        """Queries the RAG pipeline."""
        try:
            self.llm.temperature = temperature
            
            response = self.chain.invoke(question)
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def get_relevant_documents(self, question: str, k: int = 5) -> List:
        """Retrieves relevant documents."""
        try:
            docs = self.retriever.get_relevant_documents(question)
            return docs[:k]
        except Exception as e:
            st.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def update_vectorstore(self, new_vectorstore):
        """Updates the vector store."""
        self.vectorstore = new_vectorstore
        self.retriever = get_retriever(new_vectorstore)
        self.chain = self._build_chain()
