from fastapi import FastAPI

app = FastAPI(
    title="UH.portfolio"
)

@app.get("/")
def home():
    return "Welcome to Udasri Hasindu Personal info API"