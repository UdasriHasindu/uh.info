from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import rag_chain

app = FastAPI(
    title="UH.portfolio",
    description="RAG for Udasri Hasidu's Personal information"
)

# Configure CORS for your portfolio website
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://udasrihasindu.dev",
        "https://www.udasrihasindu.dev",
        "http://localhost:3000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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