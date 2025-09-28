import os
import io
import docx
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import json
import random
from datetime import datetime
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import base64
import fitz  # PyMuPDF for PDF image extraction
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# NEW IMPORTS FOR EMBEDDING FALLBACK
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain and Gemini imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# RLAgent class (copied from app.py, without Streamlit dependencies)
class RLAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.state_history = []
    def get_state_key(self, question_embedding, document_ids):
        state_key = f"{'-'.join([f'{e:.2f}' for e in question_embedding[:5]])}-{'-'.join(document_ids[:3])}"
        return state_key
    def choose_action(self, state_key, available_actions):
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    def update_q_value(self, state, action, reward, next_state=None):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ["chunk_small", "chunk_medium", "chunk_large", "similarity_standard", "similarity_mmr"]}
        max_next_q = 0
        if next_state and next_state in self.q_table:
            max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = self.q_table[state][action] + self.learning_rate * (reward + self.discount_factor * max_next_q - self.q_table[state][action])
    def save_model(self, filepath="rl_model.json"):
        with open(filepath, 'w') as f:
            json.dump(self.q_table, f)
    def load_model(self, filepath="rl_model.json"):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.q_table = json.load(f)

# Extraction and chunking helpers
def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_images_from_pdf(pdf_file):
    images = []
    try:
        pdf_file.seek(0)
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                ext = base_image["ext"]
                images.append({
                    "page": page_index + 1,
                    "index": img_index + 1,
                    "ext": ext,
                    "base64": image_base64
                })
        return images
    except Exception as e:
        return [{"error": str(e)}]

def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

# ENHANCED - FALLBACK VECTOR STORE FUNCTION
def get_vector_store(text_chunks, api_key=None):
    """Enhanced vector store creation with embedding fallback"""
    embeddings = None
    
    # Try Gemini embeddings first
    if api_key:
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001", 
                google_api_key=api_key
            )
            # Test with a small query to check quota
            test_embedding = embeddings.embed_query("test")
            print("✅ Using Gemini embeddings")
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("⚠️ Gemini quota exceeded, switching to free embeddings...")
                embeddings = None
            else:
                print(f"⚠️ Gemini error: {str(e)}")
                embeddings = None
    
    # Fallback to free Hugging Face embeddings
    if embeddings is None:
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ Using free Hugging Face embeddings (all-MiniLM-L6-v2)")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {str(e)}")
            raise e
    
    # Create vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store, embeddings

def safe_embed_query(embeddings, query):
    """Safely embed query with error handling"""
    try:
        return embeddings.embed_query(query)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print("⚠️ Quota exceeded, using fallback embedding")
            # Create simple fallback embedding
            return [0.0] * 384  # Standard embedding dimension
        else:
            raise e

def get_conversational_chain(api_key=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say 'I don't know' and do not guess or make up information. If the user asks for images from the PDF, say 'See extracted images below.' and return the images.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def is_pdf_related_question(question, doc_names):
    question_lower = question.lower()
    strictly_general_patterns = [
        "how are you", "what is your name", "tell me about yourself",
        "who are you", "what can you do", "hello", "hi ", "hey", "thanks",
        "thank you", "help me", "can you help", "i need help",
        "what's the weather", "who made you", "how do you work"
    ]
    for pattern in strictly_general_patterns:
        if question_lower.startswith(pattern):
            return False
    strong_pdf_indicators = [
        "in the document", "in the pdf", "from the pdf",
        "mentioned in", "according to", "in the text",
        "the document says", "the pdf shows", "based on the pdf",
        "what does the document", "can you find", "search for",
        "from the document", "within the pdf", "inside the document"
    ]
    for indicator in strong_pdf_indicators:
        if indicator in question_lower:
            return True
    for name in doc_names:
        if name in question_lower:
            return True
    pdf_related_words = [
        "document", "text", "content", "page", "section",
        "paragraph", "chapter", "write", "discuss", "describe",
        "explain", "analyze", "summarize", "find", "locate",
        "where", "when", "who", "what", "which", "how"
    ]
    pdf_word_count = sum(1 for word in pdf_related_words if word in question_lower)
    if pdf_word_count >= 2:
        return True
    words = question_lower.split()
    if len(words) > 6:
        general_phrases = [
            "what is", "explain to me", "tell me about",
            "how does", "why does", "define"
        ]
        is_general = any(phrase in question_lower for phrase in general_phrases)
        return not is_general
    return False

def get_direct_gemini_response(question, api_key, model_name="gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(question)
    return response.text

# Helper to extract text from images using OCR
def extract_text_from_image(image_file):
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"[OCR error: {str(e)}]"

# Load models lazily to avoid startup timeout
blip_processor = None
blip_model = None

def load_blip_models():
    global blip_processor, blip_model
    if blip_processor is None:
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return blip_processor, blip_model


def caption_image(image_file):
    try:
        processor, model = load_blip_models()  # Load only when needed
        image = Image.open(image_file).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"[Captioning error: {str(e)}]"


# FastAPI app setup
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "IRIS API is running"}
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://iris-navy.vercel.app",
        "http://localhost:3000"
    ],  # Allow deployed and local frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded docs (filename, text)
DOCUMENTS = []
RL_AGENT = RLAgent()

# Use Gemini API key from env or default
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_DEFAULT_GEMINI_API_KEY")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    texts = []
    for file in files:
        filename = file.filename.lower()
        content = await file.read()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(io.BytesIO(content))
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(io.BytesIO(content))
        else:
            continue
        DOCUMENTS.append({"filename": file.filename, "text": text})
        texts.append({"filename": file.filename, "text": text[:200]})
    return {"status": "success", "files": texts}

@app.post("/chat")
async def chat(
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    docs = []
    image_texts = []
    pdf_images = []
    image_captions = []
    
    # Add this check at the very beginning
    if files is None:
        files = []
    if files:
        for file in files:
            filename = file.filename.lower()
            content = await file.read()
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(io.BytesIO(content))
                docs.append({"filename": file.filename, "text": text})
                # Always extract images from PDF
                extracted_images = extract_images_from_pdf(io.BytesIO(content))
                pdf_images.extend(extracted_images)
                # Caption each extracted image
                for img in extracted_images:
                    if "base64" in img:
                        image_bytes = base64.b64decode(img["base64"])
                        try:
                            caption = caption_image(io.BytesIO(image_bytes))
                        except Exception as e:
                            caption = f"[Captioning error: {str(e)}]"
                        image_captions.append({"caption": caption, "page": img["page"], "index": img["index"]})
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(io.BytesIO(content))
                docs.append({"filename": file.filename, "text": text})
            elif filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                text = extract_text_from_image(io.BytesIO(content))
                image_texts.append({"filename": file.filename, "text": text})
                # Caption uploaded image
                try:
                    caption = caption_image(io.BytesIO(content))
                except Exception as e:
                    caption = f"[Captioning error: {str(e)}]"
                image_captions.append({"caption": caption, "filename": file.filename})
            else:
                continue
    else:
        docs = DOCUMENTS  # fallback to in-memory if no files sent

    use_pdf = len(docs) > 0 or len(image_texts) > 0
    if use_pdf:
        try:
            raw_text = "".join([doc["text"] for doc in docs])
            image_text = "\n".join([img["text"] for img in image_texts])
            combined_context = raw_text + "\n" + image_text
            print("Combined context sent to Gemini:", combined_context[:1000])  # Log first 1000 chars for debug
            if not combined_context.strip() and not pdf_images:
                return JSONResponse(content={"response": "No text or images could be extracted from the uploaded files or images."})
            
            # Use enhanced vector store with fallback
            text_chunks = get_text_chunks(combined_context, chunk_size=10000, chunk_overlap=1000)
            vector_store, embeddings = get_vector_store(text_chunks, api_key=GEMINI_API_KEY)
            
            # Safely embed the question
            question_embedding = safe_embed_query(embeddings, message)
            
            doc_ids = [doc["filename"][:5] for doc in docs] + [img["filename"][:5] for img in image_texts]
            state_key = RL_AGENT.get_state_key(question_embedding, doc_ids)
            chunking_actions = ["chunk_small", "chunk_medium", "chunk_large"]
            retrieval_actions = ["similarity_standard", "similarity_mmr"]
            available_actions = chunking_actions + retrieval_actions
            chosen_action = RL_AGENT.choose_action(state_key, available_actions)
            
            if chosen_action == "chunk_small":
                text_chunks = get_text_chunks(combined_context, chunk_size=5000, chunk_overlap=500)
            elif chosen_action == "chunk_medium":
                text_chunks = get_text_chunks(combined_context, chunk_size=10000, chunk_overlap=1000)
            else:
                text_chunks = get_text_chunks(combined_context, chunk_size=15000, chunk_overlap=1500)
            
            vector_store, embeddings = get_vector_store(text_chunks, api_key=GEMINI_API_KEY)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            if chosen_action in retrieval_actions:
                if chosen_action == "similarity_standard":
                    docs_retrieved = new_db.similarity_search(message)
                else:
                    docs_retrieved = new_db.max_marginal_relevance_search(message, k=4, fetch_k=10)
            else:
                docs_retrieved = new_db.similarity_search(message)
            
            chain = get_conversational_chain(api_key=GEMINI_API_KEY)
            # Use invoke instead of __call__
            response = chain.invoke({"input_documents": docs_retrieved, "question": message})
            response_output = response['output_text']
            
            if not response_output or "answer is not available in the context" in response_output.lower():
                response_output = get_direct_gemini_response(message, GEMINI_API_KEY)
            
            doc_similarity_score = 1.0 if response_output else 0.0
            response_length_score = min(len(response_output) / 1000, 1.0)
            reward = (doc_similarity_score * 0.7) + (response_length_score * 0.3)
            RL_AGENT.update_q_value(state_key, chosen_action, reward)
            RL_AGENT.save_model()
            
            return JSONResponse(content={
                "response": response_output, 
                "images": pdf_images, 
                "captions": image_captions,
                "rl_action": chosen_action,
                "reward": f"{reward:.2f}"
            })
            
        except Exception as e:
            print(f"Error in PDF processing: {str(e)}")
            # Fallback to direct Gemini response
            try:
                response_output = get_direct_gemini_response(message, GEMINI_API_KEY)
                return JSONResponse(content={
                    "response": response_output,
                    "mode": "fallback_direct"
                })
            except Exception as fallback_error:
                return JSONResponse(content={
                    "response": f"Error: {str(fallback_error)}"
                }, status_code=500)
    
    else:
        try:
            response_output = get_direct_gemini_response(message, GEMINI_API_KEY)
            return JSONResponse(content={"response": response_output})
        except Exception as e:
            return JSONResponse(content={"response": f"Error from Gemini: {str(e)}"})

@app.post("/chat-stream")
async def chat_stream(
    message: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    docs = []
    image_texts = []
    
    # Add this check at the very beginning
    if files is None:
        files = []
    if files:
        for file in files:
            filename = file.filename.lower()
            content = await file.read()
            if filename.endswith(".pdf"):
                text = extract_text_from_pdf(io.BytesIO(content))
                docs.append({"filename": file.filename, "text": text})
            elif filename.endswith(".docx"):
                text = extract_text_from_docx(io.BytesIO(content))
                docs.append({"filename": file.filename, "text": text})
            elif filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                text = extract_text_from_image(io.BytesIO(content))
                image_texts.append({"filename": file.filename, "text": text})
            else:
                continue
    else:
        docs = DOCUMENTS  # fallback to in-memory if no files sent

    use_pdf = len(docs) > 0 or len(image_texts) > 0
    if use_pdf:
        try:
            raw_text = "".join([doc["text"] for doc in docs])
            image_text = "\n".join([img["text"] for img in image_texts])
            combined_context = raw_text + "\n" + image_text
            print("Combined context sent to Gemini:", combined_context[:1000])  # Log first 1000 chars for debug
            if not combined_context.strip():
                def error_gen():
                    yield "No text could be extracted from the uploaded files or images."
                return StreamingResponse(error_gen(), media_type="text/plain")
            
            # Use enhanced vector store with fallback
            text_chunks = get_text_chunks(combined_context, chunk_size=10000, chunk_overlap=1000)
            vector_store, embeddings = get_vector_store(text_chunks, api_key=GEMINI_API_KEY)
            
            # Safely embed the question
            question_embedding = safe_embed_query(embeddings, message)
            
            doc_ids = [doc["filename"][:5] for doc in docs] + [img["filename"][:5] for img in image_texts]
            state_key = RL_AGENT.get_state_key(question_embedding, doc_ids)
            chunking_actions = ["chunk_small", "chunk_medium", "chunk_large"]
            retrieval_actions = ["similarity_standard", "similarity_mmr"]
            available_actions = chunking_actions + retrieval_actions
            chosen_action = RL_AGENT.choose_action(state_key, available_actions)
            
            if chosen_action == "chunk_small":
                text_chunks = get_text_chunks(combined_context, chunk_size=5000, chunk_overlap=500)
            elif chosen_action == "chunk_medium":
                text_chunks = get_text_chunks(combined_context, chunk_size=10000, chunk_overlap=1000)
            else:
                text_chunks = get_text_chunks(combined_context, chunk_size=15000, chunk_overlap=1500)
            
            vector_store, embeddings = get_vector_store(text_chunks, api_key=GEMINI_API_KEY)
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            
            if chosen_action in retrieval_actions:
                if chosen_action == "similarity_standard":
                    docs_retrieved = new_db.similarity_search(message)
                else:
                    docs_retrieved = new_db.max_marginal_relevance_search(message, k=4, fetch_k=10)
            else:
                docs_retrieved = new_db.similarity_search(message)
            
            chain = get_conversational_chain(api_key=GEMINI_API_KEY)
            response = chain({"input_documents": docs_retrieved, "question": message}, return_only_outputs=True)
            response_output = response['output_text']
            
            if not response_output or "answer is not available in the context" in response_output.lower():
                response_output = get_direct_gemini_response(message, GEMINI_API_KEY)
            
            doc_similarity_score = 1.0 if response_output else 0.0
            response_length_score = min(len(response_output) / 1000, 1.0)
            reward = (doc_similarity_score * 0.7) + (response_length_score * 0.3)
            RL_AGENT.update_q_value(state_key, chosen_action, reward)
            RL_AGENT.save_model()
            
            def stream_gen():
                for i in range(0, len(response_output), 20):
                    yield response_output[i:i+20]
            return StreamingResponse(stream_gen(), media_type="text/plain")
            
        except Exception as e:
            def error_gen():
                yield f"Error in processing: {str(e)}"
            return StreamingResponse(error_gen(), media_type="text/plain")
    
    else:
        try:
            response_output = get_direct_gemini_response(message, GEMINI_API_KEY)
            def stream_gen():
                for i in range(0, len(response_output), 20):
                    yield response_output[i:i+20]
            return StreamingResponse(stream_gen(), media_type="text/plain")
        except Exception as e:
            def error_gen():
                yield f"Error from Gemini: {str(e)}"
            return StreamingResponse(error_gen(), media_type="text/plain")

# Add image generation endpoint using Gemini 2.5
@app.post("/generate-image")
async def generate_image(
    data: dict = Body(...)
):
    prompt = data.get("prompt")
    size = data.get("size", "512x512")
    style = data.get("style", "natural")
    if not prompt:
        return JSONResponse(content={"error": "Prompt is required."}, status_code=400)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        image_prompt = f"""Generate a detailed description for an image based on this prompt: '{prompt}'. The description should be in the {style} style and optimized for a {size} resolution. Make the description visual and detailed so it can be understood easily."""
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(image_prompt)
        return {"description": response.text}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Add health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "embedding_fallback": "enabled"}
