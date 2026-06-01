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
collection = client.get_or_create_collection("chatbot")
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
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    if not chunks:
        return {
            "Message": "No text found in PDF"
        }
    os.remove(file.filename)
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
    retrieved_chunks = []
    if result["documents"] and result["documents"][0]:
        retrieved_chunks = result["documents"][0]
    sources = []
    if result["metadatas"] and result["metadatas"][0]:
        sources = list(
            set(
                metadata["source"]
                for metadata in result["metadatas"][0]
                if metadata and "source" in metadata
            )
        )
    if result["documents"] and result["documents"][0]:
        context = ""
        metadatas = result.get("metadatas", [[]])[0]
        for i, doc in enumerate(result["documents"][0]):
            source = "Unknown"
            if i < len(metadatas):
                metadata = metadatas[i]
                if metadata and "source" in metadata:
                    source = metadata["source"]
            context += f"\n[Source: {source}]\n{doc}\n"
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
    "Sources": sources,
    "Retrieved Chunks": retrieved_chunks,
    "Answer": final_ans
}