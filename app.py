import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import numpy as np
import random
import os
import json
import io
from PIL import Image
from datetime import datetime
import docx
import os
from dotenv import load_dotenv
load_dotenv()


# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# For direct Gemini integration
import google.generativeai as genai

# RL components
class RLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.state_history = []
        
    def get_state_key(self, question_embedding, document_ids):
        """Create a state representation based on the question and documents"""
        # Use the first 5 values of the embedding vector as a simple state representation
        # and combine with document IDs for context awareness
        state_key = f"{'-'.join([f'{e:.2f}' for e in question_embedding[:5]])}-{'-'.join(document_ids[:3])}"
        return state_key
    
    def choose_action(self, state_key, available_actions):
        """Choose between different retrieval strategies or chunk sizes based on Q-values"""
        # Exploration: try a random action
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        
        # Exploitation: choose the best known action
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state, action, reward, next_state=None):
        """Update Q-value based on reward and next state"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["chunk_small", "chunk_medium", "chunk_large", 
                                                    "similarity_standard", "similarity_mmr"]}
        
        # If next state exists, calculate the maximum Q-value for that state
        max_next_q = 0
        if next_state and next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning update formula
        self.q_table[state][action] = self.q_table[state][action] + \
                                     self.learning_rate * (reward + 
                                     self.discount_factor * max_next_q - 
                                     self.q_table[state][action])
    
    def save_model(self, filepath="rl_model.json"):
        """Save the Q-table to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f)
    
    def load_model(self, filepath="rl_model.json"):
        """Load the Q-table from a file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)

import asyncio
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_and_images(files):
    """Extract both text and images from PDF and DOCX documents"""
    text = ""
    images = []
    for file in files:
        if file.name.lower().endswith(".pdf"):
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text() + "\n"
                if '/XObject' in page['/Resources']:
                    xobjects = page['/Resources']['/XObject'].get_object()
                    for obj in xobjects:
                        if xobjects[obj]['/Subtype'] == '/Image':
                            try:
                                data = xobjects[obj].get_data()
                                image = Image.open(io.BytesIO(data))
                                images.append({
                                    "image": image,
                                    "page": page_num + 1,
                                    "filename": file.name
                                })
                            except:
                                pass
        elif file.name.lower().endswith(".docx"):
            text += extract_text_from_docx(file) + "\n"
    return text, images

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, model_name, api_key=None):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store, embeddings

def get_document_similarity(question, docs):
    """Calculate similarity score between question and retrieved documents"""
    # This is a simplified scoring method - in a real application, you would use 
    # more sophisticated metrics
    total_score = 0
    for doc in docs:
        # Count keyword matches as a simple relevance metric
        keywords = question.lower().split()
        matches = sum(1 for keyword in keywords if keyword in doc.page_content.lower())
        total_score += matches / len(keywords) if keywords else 0
    
    return total_score / len(docs) if docs else 0

def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "Google AI":
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        # Use the selected Gemini model from session state, default to gemini-2.5-pro if not set
        selected_model = st.session_state.get('gemini_model', 'gemini-2.5-pro')
        model = ChatGoogleGenerativeAI(model=selected_model, temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

def is_pdf_related_question(question, pdf_docs):
    """Determine if a question is likely related to PDFs or a general question"""
    if not pdf_docs:
        return False
    
    # Common conversational starters and general questions that should ALWAYS be general
    strictly_general_patterns = [
        "how are you", "what is your name", "tell me about yourself",
        "who are you", "what can you do", "hello", "hi ", "hey", "thanks",
        "thank you", "help me", "can you help", "I need help",
        "what's the weather", "who made you", "how do you work"
    ]
    
    # Check for strictly general patterns first
    question_lower = question.lower()
    for pattern in strictly_general_patterns:
        if question_lower.startswith(pattern):
            return False
    
    # PDF-specific indicators - if any of these are present, definitely use PDF
    strong_pdf_indicators = [
        "in the document", "in the pdf", "from the pdf",
        "mentioned in", "according to", "in the text",
        "the document says", "the pdf shows", "based on the pdf",
        "what does the document", "can you find", "search for",
        "from the document", "within the pdf", "inside the document"
    ]
    
    # Check for strong PDF indicators
    for indicator in strong_pdf_indicators:
        if indicator in question_lower:
            return True
    
    # Check if the question contains PDF file names
    pdf_names = [pdf.name.lower().replace('.pdf', '') for pdf in pdf_docs]
    for name in pdf_names:
        if name in question_lower:
            return True
    
    # For other cases, analyze the question more carefully
    # Words that suggest we might want to search in PDFs
    pdf_related_words = [
        "document", "text", "content", "page", "section",
        "paragraph", "chapter", "write", "discuss", "describe",
        "explain", "analyze", "summarize", "find", "locate",
        "where", "when", "who", "what", "which", "how"
    ]
    
    # Count how many PDF-related words are in the question
    pdf_word_count = sum(1 for word in pdf_related_words if word in question_lower)
    
    # If the question has multiple PDF-related words, prefer PDF search
    if pdf_word_count >= 2:
        return True
    
    # For longer questions (more than 6 words), prefer PDF search unless it's clearly general
    words = question_lower.split()
    if len(words) > 6:
        # Look for patterns that suggest a general question
        general_phrases = [
            "what is", "explain to me", "tell me about",
            "how does", "why does", "define"
        ]
        is_general = any(phrase in question_lower for phrase in general_phrases)
        return not is_general
    
    # For shorter questions, prefer general unless it has PDF indicators
    return False

def get_direct_gemini_response(question, api_key, model_name="gemini-1.5-flash"):
    """Get a direct response from Gemini AI without RAG"""
    genai.configure(api_key=api_key)
    # Use the selected Gemini model from session state, default to gemini-2.5-pro if not set
    selected_model = st.session_state.get('gemini_model', 'gemini-2.5-pro')
    model = genai.GenerativeModel(selected_model)
    response = model.generate_content(question)
    
    return response.text

def user_input(user_question, model_name, api_key, uploaded_files, conversation_history, rl_agent, pdf_images):
    if api_key is None:
        st.warning("Please provide API key before processing.")
        return
    
    # Get the current chat mode
    chat_mode = st.session_state.get('chat_mode', "Hybrid (Auto-detect)")
    
    # Determine if this is a PDF-related question based on mode and content
    if chat_mode == "PDF Only":
        use_pdf = True
    elif chat_mode == "General Only":
        use_pdf = False
    else:  # Hybrid mode
        use_pdf = is_pdf_related_question(user_question, uploaded_files)
        # Log the decision in the sidebar for transparency
        if use_pdf:
            st.sidebar.info("Detected as PDF-related question - using PDF search")
        else:
            st.sidebar.info("Detected as general question - using direct Gemini")
    
    # Initialize RL agent if not already exists
    if rl_agent is None:
        rl_agent = RLAgent()
        try:
            rl_agent.load_model()
            st.sidebar.success("Loaded existing RL model")
        except:
            st.sidebar.info("Initialized new RL model")
    
    # For general questions or when in General Only mode
    if not use_pdf or not uploaded_files:
        st.sidebar.info("Processing as a general question using direct Gemini response")
        try:
            response_output = get_direct_gemini_response(user_question, api_key)
            
            # Add to conversation history
            pdf_names = [file.name for file in uploaded_files] if uploaded_files else []
            conversation_history.append((
                user_question, 
                response_output, 
                "Direct " + model_name, 
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                ", ".join(pdf_names), 
                "direct_query", 
                "N/A"
            ))
            
            # Display the complete conversation history in reverse order (newest first)
            for i, (question, answer, model, timestamp, pdf_name, action, reward) in enumerate(reversed(conversation_history)):
                # User message
                user_html = f"""
                    <div class="chat-message user">
                        <div class="avatar">
                            <img src="https://i.pinimg.com/736x/3c/ae/07/3cae079ca0b9e55ec6bfc1b358c9b1e2.jpg">
                        </div>    
                        <div class="message">
                            <p>{question}</p>
                        </div>
                    </div>
                """
                st.markdown(user_html, unsafe_allow_html=True)
                
                # Bot message
                mode_info = "Mode: Direct Gemini Query" if action == "direct_query" else f"RL Agent: Strategy = {action}, Reward = {reward}"
                bot_html = f"""
                    <div class="chat-message bot">
                        <div class="avatar">
                            <img src="https://i.pinimg.com/736x/b2/8d/5d/b28d5d3c10668debab348d53802e9385.jpg">
                        </div>
                        <div class="message">
                            <p>{answer}</p>
                            <span class="mode-info">{mode_info}</span>
                        </div>
                    </div>
                """
                st.markdown(bot_html, unsafe_allow_html=True)
            
            return rl_agent
            
        except Exception as e:
            st.error(f"Error getting response from Gemini: {str(e)}")
            return rl_agent
    
    # For PDF-related questions, use the RAG system with RL
    else:
        if uploaded_files is None:
            st.warning("Please upload PDF files for document-related questions.")
            return
        
        # Process the text from PDFs
        raw_text = ""
        for file in uploaded_files:
            if file.name.lower().endswith(".pdf"):
                raw_text += get_pdf_text([file])
            elif file.name.lower().endswith(".docx"):
                raw_text += extract_text_from_docx(file)
        
        # Check if any text was extracted
        if not raw_text.strip():
            st.warning("No text could be extracted from the uploaded files. Please check your files and try again.")
            return rl_agent
        
        # Use RL agent to decide on chunking strategy
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        question_embedding = embeddings.embed_query(user_question)
        
        # Create a simple state representation
        doc_ids = [file.name[:5] for file in uploaded_files]
        state_key = rl_agent.get_state_key(question_embedding, doc_ids)
        
        # Available actions for text chunking and retrieval
        chunking_actions = ["chunk_small", "chunk_medium", "chunk_large"]
        retrieval_actions = ["similarity_standard", "similarity_mmr"]
        available_actions = chunking_actions + retrieval_actions
        
        # Choose action based on current state
        chosen_action = rl_agent.choose_action(state_key, available_actions)
        
        # Apply the chosen chunking strategy
        if chosen_action == "chunk_small":
            text_chunks = get_text_chunks(raw_text, chunk_size=5000, chunk_overlap=500)
            st.sidebar.info("RL Agent chose: Small chunks (5000 chars)")
        elif chosen_action == "chunk_medium":
            text_chunks = get_text_chunks(raw_text, chunk_size=10000, chunk_overlap=1000)
            st.sidebar.info("RL Agent chose: Medium chunks (10000 chars)")
        else:  # chunk_large
            text_chunks = get_text_chunks(raw_text, chunk_size=15000, chunk_overlap=1500)
            st.sidebar.info("RL Agent chose: Large chunks (15000 chars)")
        
        # Create vector store with the chunks
        vector_store, embeddings = get_vector_store(text_chunks, model_name, api_key)
        
        # Load the vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Apply the chosen retrieval strategy
        if chosen_action in retrieval_actions:
            if chosen_action == "similarity_standard":
                docs = new_db.similarity_search(user_question)
                st.sidebar.info("RL Agent chose: Standard similarity search")
            else:  # similarity_mmr
                docs = new_db.max_marginal_relevance_search(user_question, k=4, fetch_k=10)
                st.sidebar.info("RL Agent chose: MMR similarity search (diversity-focused)")
        else:
            # Default to standard similarity search if a chunking action was chosen
            docs = new_db.similarity_search(user_question)
        
        # Get answer from LLM
        chain = get_conversational_chain("Google AI", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        response_output = response['output_text']
        
        # Check if question is about images
        show_images = False
        image_related_terms = ["image", "picture", "photo", "figure", "diagram", "illustration", "visual"]
        for term in image_related_terms:
            if term in user_question.lower():
                show_images = True
                break
        
        # Evaluate the response quality
        doc_similarity_score = get_document_similarity(user_question, docs)
        response_length_score = min(len(response_output) / 1000, 1.0)
        
        # Combine scores
        reward = (doc_similarity_score * 0.7) + (response_length_score * 0.3)
        
        # Update the RL model
        rl_agent.update_q_value(state_key, chosen_action, reward)
        
        # Save the updated model
        rl_agent.save_model()
        
        # Add to conversation history
        pdf_names = [file.name for file in uploaded_files] if uploaded_files else []
        conversation_history.append((user_question, response_output, model_name, 
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                                    ", ".join(pdf_names), chosen_action, f"{reward:.2f}"))

        # Display the complete conversation history in reverse order (newest first)
        for i, (question, answer, model, timestamp, pdf_name, action, reward) in enumerate(reversed(conversation_history)):
            # User message
            user_html = f"""
                <div class="chat-message user">
                    <div class="avatar">
                        <img src="https://i.pinimg.com/736x/3c/ae/07/3cae079ca0b9e55ec6bfc1b358c9b1e2.jpg">
                    </div>    
                    <div class="message">
                        <p>{question}</p>
                    </div>
                </div>
            """
            st.markdown(user_html, unsafe_allow_html=True)
            
            # Bot message
            mode_info = "Mode: Direct Gemini Query" if action == "direct_query" else f"RL Agent: Strategy = {action}, Reward = {reward}"
            bot_html = f"""
                <div class="chat-message bot">
                    <div class="avatar">
                        <img src="https://i.pinimg.com/736x/b2/8d/5d/b28d5d3c10668debab348d53802e9385.jpg">
                    </div>
                    <div class="message">
                        <p>{answer}</p>
                        <span class="mode-info">{mode_info}</span>
                    </div>
                </div>
            """
            st.markdown(bot_html, unsafe_allow_html=True)
        
        # Display images if requested and available
        if show_images and pdf_images:
            st.write("### Related Images from Documents:")
            image_cols = st.columns(3)
            for i, img_data in enumerate(pdf_images[:6]):  # Show up to 6 images
                with image_cols[i % 3]:
                    st.image(img_data["image"], caption=f"From {img_data['filename']} (Page {img_data['page']})")
        
        return rl_agent

def show_rl_performance(rl_agent):
    """Display RL performance metrics and visualizations"""
    st.subheader("RL Agent Performance")
    
    # Display Q-value table
    if rl_agent and rl_agent.q_table:
        # Create a dataframe for better visualization
        q_values_data = []
        for state, actions in rl_agent.q_table.items():
            for action, value in actions.items():
                q_values_data.append({
                    "State": state[:10] + "...",  # Truncate state for display
                    "Action": action,
                    "Q-Value": value
                })
        
        if q_values_data:
            q_df = pd.DataFrame(q_values_data)
            st.write("Q-Value Table Sample:")
            st.dataframe(q_df.head(10))
            
            # Create a chart of action preferences
            action_values = q_df.groupby("Action")["Q-Value"].mean().reset_index()
            st.bar_chart(action_values.set_index("Action"))
            
            # Show best actions for different states
            best_actions = {}
            for state, actions in rl_agent.q_table.items():
                best_action = max(actions.items(), key=lambda x: x[1])[0]
                if best_action not in best_actions:
                    best_actions[best_action] = 0
                best_actions[best_action] += 1
            
            st.write("Most Commonly Preferred Actions:")
            best_actions_df = pd.DataFrame({
                "Action": list(best_actions.keys()),
                "Count": list(best_actions.values())
            })
            st.dataframe(best_actions_df)
    else:
        st.info("No RL data available yet. Interact with the chatbot to generate data.")

def main():
    st.set_page_config(page_title="IRIS", page_icon=":books:", layout="wide")
    st.header("IRIS")

    # Initialize session state variables
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'rl_agent' not in st.session_state:
        st.session_state.rl_agent = RLAgent()
        try:
            st.session_state.rl_agent.load_model()
        except:
            pass
    
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = []

    # API Key Management
    DEFAULT_API_KEY = os.getenv('GEMINI_API_KEY') # Gemini API key
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = DEFAULT_API_KEY

    # Add CSS for chat messages and mode info
    st.markdown("""
        <style>
            .chat-message {
                padding: 1.5rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
                align-items: flex-start;
            }
            .chat-message.user {
                background-color: #2b313e;
            }
            .chat-message.bot {
                background-color: #475063;
            }
            .chat-message .avatar {
                width: 15%;
                margin-right: 24px;
            }
            .chat-message .avatar img {
                max-width: 78px;
                max-height: 78px;
                border-radius: 50%;
                object-fit: cover;
                display: block;
            }
            .chat-message .message {
                width: 85%;
                padding: 0;
                color: #fff;
            }
            .chat-message .message p {
                margin: 0;
            }
            .mode-info {
                background-color: #3b4253;
                color: #fff;
                padding: 0.5rem;
                border-radius: 0.3rem;
                font-size: 0.8rem;
                margin-top: 0.5rem;
                display: block;
            }
        </style>
    """, unsafe_allow_html=True)

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Chat", "RL Analysis", "Settings"])
    
    with tab1:
        # Chat interface
        model_name = "Google AI"  # Default model
        
        # Menu and control buttons
        with st.sidebar:
            st.title("Menu:")
            
            col1, col2 = st.columns(2)
            
            reset_button = col2.button("Reset")
            clear_button = col1.button("Rerun")

            if reset_button:
                st.session_state.conversation_history = []
                st.session_state.user_question = None
                st.session_state.rl_agent = RLAgent()
                st.session_state.pdf_images = []
                st.experimental_rerun()
                
            elif clear_button:
                if 'user_question' in st.session_state:
                    st.warning("The previous query will be discarded.")
                    st.session_state.user_question = ""
                    if len(st.session_state.conversation_history) > 0:
                        st.session_state.conversation_history.pop()
                else:
                    st.warning("The question in the input will be queried again.")

            # RL parameters
            st.subheader("RL Parameters")
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            exploration_rate = st.slider("Exploration Rate", 0.01, 1.0, 0.2, 0.01)
            
            # Update RL agent parameters
            st.session_state.rl_agent.learning_rate = learning_rate
            st.session_state.rl_agent.exploration_rate = exploration_rate
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload PDF or Word Files and Click Process",
                type=["pdf", "docx"],
                accept_multiple_files=True
            )
            if st.button("Submit & Process"):
                if uploaded_files:
                    with st.spinner("Processing PDFs and extracting images..."):
                        text, images = extract_text_and_images(uploaded_files)
                        st.session_state.pdf_images = images
                        if images:
                            st.success(f"PDFs processed successfully! Extracted {len(images)} images.")
                        else:
                            st.success("PDFs processed successfully! No images found.")
                else:
                    st.warning("Please upload PDF files before processing.")

            # Mode selection
            st.subheader("Chat Mode")
            mode_options = ["Hybrid (Auto-detect)", "PDF Only", "General Only"]
            selected_mode = st.radio("Select chat mode:", mode_options)
            st.session_state.chat_mode = selected_mode

        # User input for question
        st.subheader("Ask a Question")
        user_question = st.text_input("Ask anything - about PDFs or general questions")

        if user_question:
            # Process user input based on mode
            if st.session_state.chat_mode == "PDF Only" and uploaded_files:
                st.session_state.rl_agent = user_input(
                    user_question, 
                    model_name, 
                    st.session_state.api_key, 
                    uploaded_files, 
                    st.session_state.conversation_history,
                    st.session_state.rl_agent,
                    st.session_state.pdf_images
                )
            elif st.session_state.chat_mode == "General Only":
                st.session_state.rl_agent = user_input(
                    user_question, 
                    model_name, 
                    st.session_state.api_key, 
                    None, 
                    st.session_state.conversation_history,
                    st.session_state.rl_agent,
                    []
                )
            else:
                st.session_state.rl_agent = user_input(
                    user_question, 
                    model_name, 
                    st.session_state.api_key, 
                    uploaded_files, 
                    st.session_state.conversation_history,
                    st.session_state.rl_agent,
                    st.session_state.pdf_images
                )
    
    with tab2:
        # RL analysis and visualization
        show_rl_performance(st.session_state.rl_agent)
        
    with tab3:
        # Settings and configuration
        st.subheader("Application Settings")
        
        # Model settings
        st.write("### Model Settings")
        gemini_model = st.selectbox(
            "Select Gemini Model", 
            ["gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"],
            index=0
        )
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Output Tokens", 100, 8192, 4096, 100)
        
        # Image settings
        st.write("### Image Settings")
        enable_image_generation = st.checkbox("Enable Image Generation (via Gemini)", value=False)
        
        if enable_image_generation:
            st.info("Note: Image generation will use Gemini's multimodal capabilities when available.")
            image_size = st.select_slider(
                "Image Size",
                options=["256x256", "512x512", "1024x1024"],
                value="512x512"
            )
            image_style = st.selectbox(
                "Image Style",
                ["natural", "cartoon", "artistic", "photograph", "diagram"],
                index=0
            )

        # Save settings
        if st.button("Save Settings"):
            # Store settings in session state
            st.session_state.gemini_model = gemini_model
            st.session_state.temperature = temperature
            st.session_state.max_tokens = max_tokens
            st.session_state.enable_image_generation = enable_image_generation
            st.session_state.image_size = image_size if enable_image_generation else "512x512"
            st.session_state.image_style = image_style if enable_image_generation else "natural"
            
            st.success("Settings saved successfully!")

# Add new functions for enhanced Gemini chat and image generation

def get_image_from_gemini(prompt, api_key, model="gemini-1.5-flash", size="512x512", style="natural"):
    """Generate image description using Gemini multimodal capabilities"""
    genai.configure(api_key=api_key)
    # Since Gemini doesn't directly generate images, we'll create a detailed image description
    # that could be used with other image generation services
    image_prompt = f"""Generate a detailed description for an image based on this prompt: 
    '{prompt}'. The description should be in the {style} style and optimized for a {size} resolution.
    Make the description visual and detailed so it can be understood easily."""
    # Use the selected Gemini model from session state, default to gemini-2.5-pro if not set
    selected_model = st.session_state.get('gemini_model', 'gemini-2.5-pro')
    model_obj = genai.GenerativeModel(selected_model)
    response = model_obj.generate_content(image_prompt)
    return response.text

def process_image_request(question, api_key):
    """Process requests specifically about image generation"""
    # Extract the image description from the question
    image_terms = ["create an image", "generate an image", "make an image", 
                   "draw", "create a picture", "generate a picture", "design an image"]
    
    contains_image_request = any(term in question.lower() for term in image_terms)
    
    if contains_image_request:
        # Extract what comes after the image request term
        image_prompt = ""
        for term in image_terms:
            if term in question.lower():
                parts = question.lower().split(term, 1)
                if len(parts) > 1:
                    image_prompt = parts[1].strip()
                    break
        
        if not image_prompt:
            image_prompt = question  # Use the whole question if we couldn't extract a specific part
        
        # Get the image description
        image_description = get_image_from_gemini(
            image_prompt, 
            api_key,
            model=st.session_state.get('gemini_model', 'gemini-1.5-flash'),
            size=st.session_state.get('image_size', '512x512'),
            style=st.session_state.get('image_style', 'natural')
        )
        
        return True, image_description
    
    return False, None

def enhanced_direct_gemini_response(question, api_key, model_name="gemini-1.5-flash", temperature=0.3, max_tokens=4096):
    """Enhanced direct response from Gemini AI with better configuration"""
    genai.configure(api_key=api_key)
    # First check if this is an image generation request
    is_image_request, image_content = process_image_request(question, api_key)
    if is_image_request:
        return f"""I've created an image description based on your request:
        ---
        {image_content}
        ---
        Note: Since I can't directly generate images, I've provided a detailed description that could be used with image generation tools."""
    # For regular text queries
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": max_tokens,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]
    # Use the selected Gemini model from session state, default to gemini-2.5-pro if not set
    selected_model = st.session_state.get('gemini_model', 'gemini-2.5-pro')
    model = genai.GenerativeModel(
        selected_model,
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    response = model.generate_content(question)
    return response.text

if __name__ == "__main__":
    main()