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

load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = 'English'

def get_prompt_template(language):
    """Get the appropriate prompt template based on selected language"""
    base_template = """
        You are a knowledgeable and friendly product expert who understands both technical terms and everyday language. 
        Interpret and respond to queries using common terms while providing accurate technical information.

        TERM MAPPING (interpret these terms as equivalent):
        - Mileage/Average/Fuel Economy/Gas Consumption
        - Speed/Performance/Pick-up
        - Horse Power/Power/Strength/Pulling Power
        - Price/Cost/Value/Worth
        - Size/Dimensions/Space
        - Features/Additions/Extras
        - New/Latest/Recent
        - Used/Second-hand/Pre-owned
        - Automatic/Self-driving/Auto
        - Manual/Hand-operated/Stick-shift
        - Electric/Battery/EV
        - Hybrid/Mixed/Dual
        - Problems/Issues/Complaints
        - Warranty/Guarantee/Coverage
        - Maintenance/Service/Upkeep
        
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
            *Performance: [Key performance details]
            *Features: [Most important feature(s)]
            *Technical Specifications: [Essential technical details only]
            *Price: [Price, if available]
           - Use bullet points for clear differentiation

        4. when user ask which product should i buy:

        - don't show available products part:

           Analysis Matrix:
           only give this column of available product only.
            | Criteria          | Product A | Product B | 
            |-------------------|-----------|-----------|
            | Performance       |           |           |           
            | Features          |           |           |           
            | Price             |           |           |           
            | User Rating       |           |           |           

            Recommended: [Product Name]

            Key Advantages:
            1. Performance Benefits:
            • [Benefit 1]
            • [Benefit 2]

            2. Standout Features:
            • [Feature 1]
            • [Feature 2]

            3. Value Proposition:
            • [Value point 1]
            • [Value point 2]

            Justification:
            [Clear explanation of why this product is recommended]

            Alternative Recommendations:
            - Budget Option: [Product Name]
            - Premium Option: [Product Name]
            - Specialized Use: [Product Name]

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
        """
    
    # Add language-specific ending
    if language == 'Hindi':
        return base_template + '\nProvide a direct and precise response in normal hindi following the guidelines above: """'
    else:
        return base_template + '\nProvide a direct and precise response in english following the guidelines above: """'

def initialize_chatbot():
    """Initialize the chatbot with the latest OpenAI models"""
    try:
        llm = OpenAI(
            model=os.getenv("gptmodel"),
            temperature=0.1,
            api_key=os.environ["OPENAI_API_KEY"]
        )
        
        embed_model = OpenAIEmbedding(
            model=os.getenv("embmodel"),
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

def create_query_engine(index, language='English'):
    """Create a query engine with enhanced product comparison and recommendation capabilities"""
    template = get_prompt_template(language)
    qa_prompt = PromptTemplate(template)
    
    return index.as_query_engine(
        text_qa_template=qa_prompt,
        similarity_top_k=7,
        response_mode="compact"
    )

def get_response_text(response):
    """Extract just the response text from the LlamaIndex response object"""
    return str(response.response)

def display_question_boxes():
    """Display clickable question boxes in a grid layout"""
    st.markdown("### Quick Questions")
    
    # Define common questions about products
    questions = [
        {"title": "Metro Hunter", "query": "answer metro hunter specification in precise?"},
        {"title": "Retro Hunter", "query": "answer retro hunter specification in precise?"},
        {"title": "specification of bike", "query": "compare the bikes in precise way?"},
        {"title": "Compare Products", "query": "compare the bikes in precise way"},
    ]
    
    # Create a 2x3 grid layout
    cols = st.columns(3)
    for idx, question in enumerate(questions):
        with cols[idx % 3]:
            if st.button(
                question["title"],
                key=f"q_{idx}",
                use_container_width=True,
                help=question["query"]
            ):
                return question["query"]
    
    return None

def main():
    st.set_page_config(
        page_title="Smart Product Information System",
        page_icon="🏪",
        layout="wide"
    )
    
    st.title("Smart Product Information System 🏪")
    
    # Language selector in sidebar
    with st.sidebar:
        st.header("Settings")
        selected_language = st.selectbox(
            "Choose Language",
            ["English", "Hindi"],
            key="language_selector"
        )
        
        # Update session state and reinitialize query engine if language changes
        if selected_language != st.session_state.selected_language:
            st.session_state.selected_language = selected_language
            if 'query_engine' in st.session_state:
                del st.session_state.query_engine
    
    # Initialize system
    try:
        if 'query_engine' not in st.session_state:
            with st.spinner("Initializing system..."):
                index = initialize_chatbot()
                st.session_state.query_engine = create_query_engine(index, st.session_state.selected_language)
            st.success("System initialized successfully!")
        
        # Sidebar with instructions
        with st.sidebar:
            st.header("How to Use")
            st.markdown("""
            You can:
            1. Click on the question boxes above
            2. Ask your own questions in the chat
            3. Get detailed product information
            4. Compare multiple products
            
            Example questions:
            - "What cars are available?"
            - "Tell me about the Toyota Camry"
            """)
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display question boxes at the top
        selected_query = display_question_boxes()
        
        # Add some space between boxes and chat
        st.markdown("---")
        
        # Main chat interface
        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.chat_message("user").write(content)
            else:
                st.chat_message("assistant").markdown(content)
        
        # Process selected query from question boxes
        if selected_query:
            try:
                with st.spinner("Analyzing your question..."):
                    response = st.session_state.query_engine.query(selected_query)
                    response_text = get_response_text(response)
                
                # Display the response
                st.chat_message("user").write(selected_query)
                st.chat_message("assistant").markdown(response_text)
                
                # Update chat history
                st.session_state.chat_history.extend([
                    {"role": "user", "content": selected_query},
                    {"role": "assistant", "content": response_text}
                ])
                
            except Exception as e:
                st.error(f"Error processing question: {str(e)}")
                st.info("Please try rephrasing your question.")
        
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
