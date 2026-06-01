# PDF RAG Chatbot with FastAPI, ChromaDB & Gemini
##This Project is Live now
Check it out here : pdf-rag-with-conversation-memory-production.up.railway.app/docs

## Overview

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their contents.

Instead of relying solely on the LLM's knowledge, the chatbot retrieves relevant information from uploaded documents using semantic search and provides grounded answers based on the retrieved context.

The project also includes conversation memory, query rewriting, and answer verification using a critic agent.

---

## Features

* Upload PDF documents
* Extract text from PDFs
* Automatic document chunking
* Generate embeddings using Sentence Transformers
* Store vectors in ChromaDB
* Semantic search for relevant context retrieval
* Gemini-powered answer generation
* Query rewriting for follow-up questions
* Conversation memory
* Critic agent for answer verification
* Persistent vector database storage

---

## Tech Stack

### Backend

* Python
* FastAPI

### AI & NLP

* Google Gemini API
* Sentence Transformers (`all-MiniLM-L6-v2`)

### Vector Database

* ChromaDB

### Document Processing

* PyPDF

### Data Validation

* Pydantic

---

## Project Architecture

```text
PDF Upload
    │
    ▼
Text Extraction (PyPDF)
    │
    ▼
Chunking
    │
    ▼
Embeddings (Sentence Transformers)
    │
    ▼
ChromaDB Vector Store
    │
    ▼
Semantic Search
    │
    ▼
Relevant Context
    │
    ▼
Query Rewriter Agent
    │
    ▼
Answer Agent (Gemini)
    │
    ▼
Critic Agent
    │
    ▼
Final Response
```

---

## API Endpoints

### Upload PDF

```http
POST /upload
```

Uploads a PDF, extracts text, generates embeddings, and stores chunks in ChromaDB.

### Chat with Document

```http
POST /chat
```

Example Request:

```json
{
  "question": "What is RAM?"
}
```

Example Response:

```json
{
  "Query": "What is RAM?",
  "Answer": "RAM stands for Random Access Memory..."
}
```

---

## How It Works

### 1. PDF Processing

When a PDF is uploaded:

* Text is extracted using PyPDF.
* The text is split into chunks.
* Each chunk is converted into embeddings.
* Embeddings are stored in ChromaDB.

### 2. Retrieval

When a user asks a question:

* The question is rewritten if necessary using conversation history.
* The rewritten query is converted into an embedding.
* ChromaDB performs semantic similarity search.
* Relevant chunks are retrieved.

### 3. Generation

The retrieved context and user question are sent to Gemini.

Gemini generates an answer grounded in the retrieved document content.

### 4. Verification

A critic agent reviews the generated answer and improves it if necessary to ensure it remains grounded in the retrieved context.

---

## Installation

### Clone Repository

```bash
git clone https://github.com/iamindra07/pdf-rag-with-conversation-memory.git

cd pdf-rag-with-conversation-memory
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Run the Application

```bash
uvicorn main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Learning Outcomes

This project helped me understand:

* Retrieval-Augmented Generation (RAG)
* Embeddings and vector representations
* Semantic search
* Vector databases
* FastAPI backend development
* Prompt engineering
* Multi-agent workflows
* Document question answering systems

---

## Future Improvements

* Multi-document support
* Chat session management
* Source citation highlighting
* Better chunking strategies
* Hybrid search (keyword + semantic)
* Web interface using React

---

## Author

Indranil Majumder

Backend Developer | Python | FastAPI | AI Applications
