from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import chatbot  # Import the chatbot function from Step 2

app = FastAPI()

# Configure CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data model for incoming requests
class QueryModel(BaseModel):
    query: str

@app.post("/chatbot")
async def chat(query_model: QueryModel):
    query = query_model.query
    if query:
        answer = chatbot(query)
        return {"answer": answer}
    else:
        return {"answer": "No query provided."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
