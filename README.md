# 🔮 Gemini RAG Chatbot with Reinforcement Learning (RL)

This project is a **powerful AI-driven chatbot** application built with **Google's Gemini Pro API**, **LangChain**, **Streamlit**, and **Reinforcement Learning (RL)** to improve the **contextual accuracy** of answers over time.

It supports **PDF and DOCX document ingestion**, **image extraction**, **RAG-based question answering**, **multimodal (image + text) chat**, and **dynamic tuning** of retrieval strategy using **Q-Learning**.

---

## 🚀 Features

### 📚 Document-Aware Chat
- Upload multiple **PDF** or **DOCX** files
- Automatically extracts and chunks text + images
- Ask questions directly about the uploaded documents

### 🤖 Gemini Chat Modes
- **Direct Gemini Chat**: Uses Gemini without context
- **Gemini + RAG**: Uses LangChain for document context retrieval before answering

### 🧠 Reinforcement Learning Agent
- Uses **Q-learning** to improve retrieval strategy and chunking over time
- Adjusts actions based on rewards from Gemini's helpfulness

### 🖼️ Multimodal AI Support
- Extracts **images from PDFs**
- Generates **Gemini Vision responses** for images

### 📈 Analytics + Customization
- View RL Agent Q-Table
- Customize Gemini parameters (model, temperature, tokens, etc.)
- Switch between `similarity_search` and `mmr_search`

---

## 📸 Screenshots

| Upload Docs | Chat View | RL Analysis |
|-------------|-----------|-------------|
| ![upload](./assets/upload_docs.png) | ![chat](./assets/chat_view.png) | ![rl](./assets/rl_table.png) |

---

## ⚙️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Language Models**: [Gemini Pro / Vision](https://deepmind.google/technologies/gemini/)
- **Retrieval System**: [LangChain + FAISS](https://python.langchain.com/)
- **Reinforcement Learning**: Q-Learning Agent (custom)
- **PDF/DOCX Parsing**: PyMuPDF (`fitz`), `docx`
- **Image Processing**: PIL, base64
- **Vector Store**: FAISS

---

## 🧠 How RL Improves Performance

The system explores and learns the best combination of:
- Chunk size: `small` or `large`
- Search type: `similarity` vs `mmr` (max marginal relevance)

Each user interaction updates the Q-table based on feedback from Gemini, creating a **self-optimizing chatbot** over time.

---

## 🔧 Setup & Installation

### 1. Clone the Repo
#git clone https://github.com/yourusername/gemini-rag-rl-chatbot.git
#cd rag-chat
#python -m venv myenv
#myenv/scripts/activate
#pip install -r requirements.txt
# create and .env file 
GEMINI_API_KEY=" add you API "
#pip install google-generativeai
#streamlit run app.py
