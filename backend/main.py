from fastapi import FastAPI

app = FastAPI(title="NEURYX")

@app.get("/")
def health_check():
    return {"status": "ok"}
