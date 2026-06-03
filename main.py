from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
import chromadb
import os
app = FastAPI()
load_dotenv()
ai1 = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
ai2 = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("chatbot")
#variable to store chat history
chat_history = []
all_chunks = []
all_metadata = []
bm25 = None

#query format
class Request(BaseModel):
    question : str

def build_bm25():
    global bm25

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

def bm25_search(query, top_k):
    global bm25

    if bm25 is None:
        return []

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_results = sorted(
        zip(all_chunks, all_metadata, scores),
        key=lambda x: x[2],
        reverse=True
    )

    return ranked_results[:top_k]


#function to rewrite the query based on the conversation history
def rewrite_query(question: str):
    history_text = ""

    for msg in chat_history[-4:]:
        history_text += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
    Conversation History:
    {history_text}

    Current Question:
    {question}

    Rewrite the question so it is completely standalone.
    Return only the rewritten question.
    """
    
    try:
        result = ai1.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return result.text
    except Exception as e:
        print(f"Gemini failed: {e}")
    try:
        result = ai2.responses.create(
            input=prompt,
            model="openai/gpt-oss-20b",
        )
        return result.output_text
    except Exception as e:
        print(f"OpenAI failed: {e}")
    return question

#function to generate answer based on the context
def answer_agent(context:str, question:str):
    history_text = ""

    for msg in chat_history[-6:]:
        history_text += f"{msg['role']}: {msg['content']}\n"
    
    prompt = f"""
        Conversation History:
        {history_text}

        Context:
        {context}

        Question:
        {question}

        Answer ONLY using the provided context.

        If the answer is not present in the context,
        say:
        "I could not find this information in the uploaded documents."

        Do not use outside knowledge.
    """
    try:
        result = ai1.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        answer = result.text
        return answer
    except Exception as e:
        print(f"Gemini failed: {e}")
    try:
        result = ai2.responses.create(
            input=prompt,
            model="openai/gpt-oss-20b",
        )
        answer = result.output_text
        return answer
    except Exception as e:
        print(f"OpenAI failed: {e}")
    return "All AI providers are currently unavailable."

#function to validate the generated answer
def critic_agent(context:str, question:str,ans:str):
    history_text = ""
    for msg in chat_history[-6:]:
        history_text += f"{msg['role']}: {msg['content']}\n"
    prompt = f"""
        Verify whether the answer is fully supported by the provided context.
        If any statement is not supported by the context, remove it. Return only the corrected answer.
        [NOTE: JUST RETURN THE FINAL ANSWER NO NEED TO SAY YOU IMPROVED IT OR THE ANSWER IS ALREADY GROUNDED.
        JUST RETURN THE PROPER CORRECT ANSWER ONLY]
        
        conversation history:
        {history_text}
        
        context:
        {context}
        
        question:
        {question}
        
        answer:
        {ans}
    """
    try:
        result = ai1.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        answer = result.text
        return answer
    except Exception as e:
        print(f"Gemini failed: {e}")
    try:
        result = ai2.responses.create(
            input=prompt,
            model="openai/gpt-oss-20b",
        )
        answer = result.output_text
        return answer
    except Exception as e:
        print(f"OpenAI failed: {e}")
    return ans


@app.post("/upload")
async def upload (file:UploadFile = File(...)):
    with open (file.filename,"wb")as f:
        content = await file.read()
        f.write(content)
    
    reader = PdfReader(file.filename)
    text = ""
    for page in reader.pages:
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    os.remove(file.filename)
    if not chunks:
        return {
            "Message": "No text found in PDF"
        }
    embeddings = model.encode(chunks).tolist()
    try:
        existing = collection.get(
            where={"source": file.filename}
        )
        if existing["ids"]:
            collection.delete(
                ids=existing["ids"]
            )
    except:
        pass
    for i , chunk in enumerate(chunks):
        collection.add(
            ids=[f"{file.filename}_{i}"],
            documents=[chunk],
            embeddings=[embeddings[i]],
            metadatas=[
                {
                    "source":file.filename
                }
            ]
        )
        all_chunks.append(chunk)
        all_metadata.append(
            {
                "source": file.filename
            }
        )
    build_bm25()
    return {
        "Message":"File uploaded successfully",
        "Total chunks":len(chunks)
    }



@app.post("/chat")
def chat(req:Request):
    query = req.question
    rewritten_query = rewrite_query(query)
    query_embedding = model.encode([rewritten_query]).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=6
    )
    bm25_results = bm25_search(
    rewritten_query,
    top_k=3
    )
    vector_results = []

    for doc, metadata in zip(
        result["documents"][0],
        result["metadatas"][0]
    ):
        vector_results.append(
            (   
                doc,
                metadata,
                1.0
            )
        )
    combined = vector_results + bm25_results
    seen = set()
    unique_results = []

    for doc, metadata, score in combined:

        if doc not in seen:
            seen.add(doc)

            unique_results.append(
                (
                    doc,
                    metadata
                )
            )
    context = ""

    for doc, metadata in unique_results:

        source = "unknown"

        if metadata:
            source = metadata.get(
                "source",
                "unknown"
            )

        context += (
            f"\n[Source: {source}]\n"
            f"{doc}\n"
        )
    retrieved_chunks = [
        doc
        for doc, metadata
        in unique_results
    ]
    ans = answer_agent(context,query)
    final_ans = critic_agent(context,query,ans)
    
    chat_history.append({
    "role":"user",
    "content":query
    })
    chat_history.append({
    "role":"assistant",
    "content":final_ans
    })
    chat_history[:] = chat_history[-10:]
    return {
    "Original Query": query,
    "Rewritten Query": rewritten_query,
    "Sources": source,
    "Retrieved Chunks": retrieved_chunks,
    "Answer": final_ans
}