# ingestion.py
import os
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_pdfs():
    """Ingest PDFs with fixed ChromaDB configuration"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize with latest models
        llm = OpenAI(
            model="gpt-4-0125-preview",
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        # Set global settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Load PDFs
        logger.info("Loading PDFs from data directory...")
        documents = SimpleDirectoryReader(
            input_dir="data",
            recursive=True
        ).load_data()
        
        # Create ChromaDB client - Fixed configuration
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create collection
        chroma_collection = client.get_or_create_collection(name="quickstart_gpt4")
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        logger.info("Creating index from documents...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info("PDF ingestion completed successfully!")
        return index
        
    except Exception as e:
        logger.error(f"Error during PDF ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("\n=== PDF Ingestion System ===")
        print("Starting ingestion process...\n")
        ingest_pdfs()
        print("\nIngestion completed successfully!")
        
    except Exception as e:
        print(f"\nFatal Error: {str(e)}")
        print("Please check your configuration and try again.")