from fastapi import FastAPI
from pydantic import BaseModel
from rag import rag_chain

app = FastAPI(
    title="UH.portfolio",
    description="RAG for Udasri Hasidu's Personal information"
)

@app.get("/")
def home():
    return "Welcome to Udasri Hasindu Personal info API"

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_questions(request: QuestionRequest):

    response = rag_chain.invoke(request.question)

    return{
        "Question": request.question,
        "Answer": response
    }