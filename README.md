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

<img width="1919" height="972" alt="image" src="https://github.com/user-attachments/assets/452d6989-a115-486e-b15d-81866744de7f" />
<img width="1901" height="969" alt="image" src="https://github.com/user-attachments/assets/36916145-ee7a-45b1-8f30-c88cbcb63b34" />



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


## 🔧 Setup & Installation

Follow these steps to get the project up and running locally:

### 1. 🚀 Clone the Repository

```bash
git clone https://github.com/yourusername/gemini-rag-rl-chatbot.git
cd gemini-rag-rl-chatbot
```

### 2. 🧪 Set Up a Virtual Environment (Windows)

```bash
python -m venv myenv
myenv\Scripts\activate
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. 🔐 Configure Environment Variables

Create a `.env` file in the root directory and add your [Gemini API Key](https://makersuite.google.com/app/apikey):

```env
GEMINI_API_KEY="your_gemini_api_key_here"
```

### 5. 📥 Install Gemini SDK

```bash
pip install google-generativeai
```

### 6. 🧠 Run the App

```bash
streamlit run app.py
```

---



