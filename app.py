# streamlit_app.py
import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import PromptTemplate
import chromadb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def initialize_chatbot():
    """Initialize the chatbot with the latest OpenAI models"""
    try:
        load_dotenv()
        
        llm = OpenAI(
            model="gpt-4-0125-preview",
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        load_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = load_client.get_collection("quickstart_gpt4")
        
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        
        return index
    
    except Exception as e:
        logger.error(f"Error initializing chatbot: {str(e)}")
        raise

def create_query_engine(index):
    """Create a query engine with enhanced product comparison and recommendation capabilities"""
    template = (
        """You are a knowledgeable product expert. Provide precise, direct information about products.
        
        RESPONSE GUIDELINES:

        1. Category Listings:
           - Start with "Available products:"
           - Present as a numbered list
           - End with "Which product interests you?"
           Example:
           "Available products:
           1. Model X Performance
           2. Model Y Long Range
           3. Model 3 Standard
           Which product interests you?"

        2. Product Descriptions:
           - Start with "[Product Name] Specifications:"
           - Group information under clear headings:
             * Performance
             * Features
             * Technical Specifications
             * Price (if available)

        3. Product Comparisons:
           - Start with "Comparing [Product A] vs [Product B]:"
           - Organize by categories:
             * Performance Differences
             * Feature Comparison
             * Technical Specifications
             * Price Comparison
           - Use bullet points for clear differentiation

        4. Product Recommendations:
           - Start with "Recommended: [Product Name]"
           - List key advantages:
             * Performance Benefits
             * Standout Features
             * Value Proposition
           - Conclude with clear justification

        5. Missing Information:
           - Specify exactly which aspects are unavailable
           - Focus on available specifications
        
        6. For Specific Attribute Queries (like torque, power, mileage, etc.):
           1. IF question asks about an attribute (torque, power, etc.) for a category (cars, bikes, etc.):
           - List that attribute for ALL products in that category
           - Format as a numbered list
           Example for "What is the torque of all cars?":
           "Torque specifications:
           1. Toyota Fortuner: 500 Nm
           2. Hyundai Creta: 250 Nm
           3. Tata Harrier: 350 Nm"

           2. IF question asks about an attribute for a specific product:
           - Return ONLY that product's attribute value
           - Format: "[Product Name]: [Attribute Value]"
           Example: "Toyota Fortuner: 500 Nm torque"

        Current Question: {query_str}
        
        Context: {context_str}
        
        Provide a direct response following the guidelines above: """
    )
    
    qa_prompt = PromptTemplate(template)
    
    return index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=7,
        response_mode="compact"
    )

def get_response_text(response):
    """Extract just the response text from the LlamaIndex response object"""
    return str(response.response)

def main():
    st.set_page_config(
        page_title="Smart Product Information System",
        page_icon="🏪",
        layout="wide"
    )
    
    st.title("Smart Product Information System 🏪")
    
    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                index = initialize_chatbot()
                st.session_state.query_engine = create_query_engine(index)
            st.success("System initialized successfully!")
        
        # Sidebar with instructions
        with st.sidebar:
            st.header("How to Use")
            st.markdown("""
            You can:
            1. Ask about available products in a category
            2. Get detailed information about specific products
            3. Compare multiple products
            4. Ask for product recommendations
            
            Example questions:
            - "What cars are available?"
            - "Tell me about the Toyota Camry"
            - "Compare Honda Civic vs Toyota Corolla"
            - "What's the best car under $30,000?"
            """)
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Main chat interface
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").markdown(content)
        
        # Chat input
        if query := st.chat_input("Ask about products..."):
            st.chat_message("user").write(query)
            
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
    
    except Exception as e:
        st.error(f"System Error: {str(e)}")
        st.warning("Please check your configuration and try again.")

if __name__ == "__main__":
    main()