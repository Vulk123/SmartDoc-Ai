 🚀 SmartDoc AI — RAG-based LLM Assistant

SmartDoc AI is a Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions. The system retrieves relevant context from the documents and generates accurate, grounded answers using LLMs.


 🔥 Features

* 📄 Upload PDF / TXT / MD documents
* ✂️ Automatic text chunking
* 🧠 Embedding generation using OpenAI
* 🔍 Semantic search with FAISS vector database
* 🤖 Context-aware LLM responses (no hallucination)
* ⚡ FastAPI backend with REST APIs
* 🌐 Swagger UI for testing endpoints



🧠 Tech Stack

* Python
* LangChain
* FAISS (Vector Database)
* OpenAI API
* FastAPI



📁 Project Structure


smartdoc-ai/
│
├── main.py              # FastAPI backend
├── rag_pipeline.py      # RAG logic (ingestion + query)
├── uploads/             # Uploaded documents
├── README.md
└── requirements.txt




⚙️ Installation

1. Clone the repository

```
git clone https://github.com/your-username/smartdoc-ai.git
cd smartdoc-ai
```

 2. Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Set OpenAI API Key

```
setx OPENAI_API_KEY "your_api_key_here"
```

---
 ▶️ Run the Application

```
python -m uvicorn main:app --reload
```

---

 🌐 API Endpoints

Open Swagger UI:

```
http://127.0.0.1:8000/docs
```

 Key Endpoints:

* `POST /upload` → Upload documents
* `POST /query` → Ask questions
* `GET /health` → Check API status

---

🧪 How It Works

1. Upload document → chunking + embeddings
2. Store in FAISS vector database
3. Query → retrieve relevant chunks
4. LLM generates answer using context

---

## 📌 Example Query

```json
{
  "session_id": "your_session_id",
  "question": "Summarize the document"
}
```

---

🎯 Why This Project

This project demonstrates:

* Strong understanding of RAG architecture
* LLM integration in real-world systems
* Vector databases and semantic search
* End-to-end AI pipeline development

---

 🚀 Future Improvements

* Chat UI frontend (React / Streamlit)
* Multi-document querying
* Streaming responses
* Chat memory integration

---

 👨‍💻 Author

Gargi Rathore

---

⭐ If you like this project

Give it a star ⭐ on GitHub!
