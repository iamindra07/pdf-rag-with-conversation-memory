from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
import os
app = FastAPI()
load_dotenv()
ai = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("chatbot")
#variable to store chat history
chat_history = []

#query format
class Request(BaseModel):
    question : str

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
        result = ai.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )
        return result.text.strip()

    except:
        return question

#function to generate answer based on the context
def answer_agent(context:str, question:str):
    history_text = ""

    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    
    prompt = f"""
        Conversation History:
        {history_text}

        Context:
        {context}

        Question:
        {question}

        Answer using the context.
    """
    try:
        result = ai.models.generate_content(
            model="gemini-3.5-flash",
            contents= prompt
        )
        answer = result.text
    except Exception as e:
        answer = str(e)
    return answer

#function to validate the generated answer
def critic_agent(context:str, question:str,ans:str):
    history_text = ""
    for msg in chat_history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    prompt = f"""
        using the conversation history(if any), context and answer check the answer is grounded in the context.
        if weak, improve it. 
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
        result = ai.models.generate_content(
            model="gemini-3.5-flash",
            contents= prompt
        )
        answer = result.text
    except Exception as e:
        answer = str(e)
    return answer
    


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
    chunks = [text[i:i+100] for i in range(0, len(text), 100)]
    embeddings = model.encode(chunks).tolist()
    for i , chunk in enumerate(chunks):
        collection.add(
            ids=[f"{file.filename}_{i}"],
            documents=[chunk],
            embeddings=[embeddings[i]]
        )
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
        n_results=4
    )
    if result["documents"] and result["documents"][0]:
        context = "\n".join(result["documents"][0])
    else:
        context = "No relevant context found."
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
    "Retrieved Chunks": result["documents"][0],
    "Answer": final_ans
    }