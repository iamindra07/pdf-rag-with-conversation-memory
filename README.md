# Multi-Provider RAG Chatbot with Conversation Memory

A Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, ChromaDB, Sentence Transformers, Google Gemini, and Groq. The system supports multi-document retrieval, conversation memory, query rewriting, answer validation, and automatic model fallback.

# This Project is Live now

[Try the Application](https://pdf-rag-with-conversation-memory-production.up.railway.app/docs)


## Features

* Upload and chat with multiple PDF documents
* Semantic search using Sentence Transformers
* Vector storage with ChromaDB
* Conversation memory for follow-up questions
* Query rewriting for context-aware retrieval
* Answer validation using a critic agent
* Multi-provider AI fallback system

  * Google Gemini
  * Groq (GPT-OSS-20B)
* Source attribution for retrieved information
* Persistent vector database storage
* REST API built with FastAPI

---

## Architecture

User Question
↓
Query Rewriter Agent
↓
Embedding Generation
↓
ChromaDB Vector Search
↓
Context Retrieval
↓
Answer Generation Agent
↓
Critic Agent
↓
Final Grounded Answer

---

## Tech Stack

### Backend

* Python
* FastAPI
* Pydantic

### AI & NLP

* Google Gemini API
* Groq API
* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Database

* ChromaDB

### Document Processing

* PyPDF

---

## API Endpoints

### Upload PDF

POST /upload

Upload a PDF document and store its embeddings.

Response:

```json
{
  "Message": "File uploaded successfully",
  "Total chunks": 25
}
```

### Chat with Documents

POST /chat

Request:

```json
{
  "question": "What is RAM?"
}
```

Response:

```json
{
  "Original Query": "What is RAM?",
  "Rewritten Query": "What is Random Access Memory (RAM)?",
  "Sources": [
    "computer_fundamentals.pdf"
  ],
  "Answer": "RAM is..."
}
```

---

## Key Components

### Query Rewriter Agent

Converts follow-up questions into standalone questions.

Example:

User:

```text
What is RAM?
```

User:

```text
Why do we need it?
```

Rewritten Query:

```text
Why do we need Random Access Memory (RAM)?
```

---

### Retrieval System

* Converts documents into embeddings
* Stores vectors in ChromaDB
* Retrieves the most relevant chunks for a query

---

### Answer Agent

Generates answers strictly from retrieved document context.

If information is not available:

```text
I could not find this information in the uploaded documents.
```

---

### Critic Agent

Validates generated answers and removes unsupported claims to reduce hallucinations.

---

### Multi-Provider Fallback

The application automatically switches providers if the primary model fails.

Flow:

```text
Gemini
   ↓
Groq
   ↓
Fallback Response
```

This improves reliability when:

* API quotas are exceeded
* Rate limits occur
* A provider becomes unavailable

---

## Installation

Clone the repository:

```bash
git clone https://github.com/iamindra07/Multi-Provider-RAG-Chatbot-with-Conversation-Memory.git
cd Multi-Provider-RAG-Chatbot-with-Conversation-Memory
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

---

## Run the Application

```bash
uvicorn main:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Project Highlights

* Built a complete RAG pipeline from scratch
* Implemented multi-document retrieval
* Added conversation memory for contextual chats
* Added query rewriting for improved retrieval accuracy
* Implemented answer verification through a critic agent
* Added automatic AI provider failover
* Deployed as a production-ready FastAPI service

---

## Future Improvements

* Overlapping chunking
* Hybrid search (Keyword + Vector)
* Document management APIs
* Streaming responses
* Web-based frontend
* Reranking models
* User authentication

---

## Author

Indranil Majumder

GitHub:
https://github.com/iamindra07

LinkedIn:
https://www.linkedin.com/in/indranil-majumder-0086243a1/

---

## License

This project is open-source and available under the MIT License.
