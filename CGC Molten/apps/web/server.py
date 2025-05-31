from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from core.retrieval.retrieval import search_and_retrieve
from core.chat.chat_chain import create_chat_chain
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import pandas as pd
import ast
from config.config import EMBEDDINGS_FILE, DEVICE, TOP_K, CORS_ALLOWED_ORIGINS, MODEL_NAME
from fastapi.middleware.cors import CORSMiddleware
from config.config import CORS_ALLOWED_ORIGINS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # <- THIS needs to be the actual middleware class
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="apps/web/static"))


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_file = Path("static/index.html")
    return index_file.read_text(encoding="utf-8")

def load_data(file_path: str, device: str = DEVICE):
    df = pd.read_csv(file_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(device)
    return pages_and_chunks, embeddings_tensor

pages_and_chunks, embeddings = load_data(EMBEDDINGS_FILE, device=DEVICE)
embedding_model = SentenceTransformer(model_name_or_path=MODEL_NAME, device=DEVICE)
embedding_model.to(DEVICE)
chain = create_chat_chain()

class QueryRequest(BaseModel):
    question: str
    history: str = ""  

@app.post("/query/")
async def get_response(request: QueryRequest):
    context = search_and_retrieve(
        request.question, embedding_model, pages_and_chunks, embeddings, top_k=TOP_K
    )
    if not context:
        return {
            "response": "Sorry, I couldn't find relevant information.",
            "history": request.history,
            "source": "Unknown Source",
            "sources": []
        }
    
    combined_context = "\n\n".join([c["text"] for c in context[:TOP_K]])
    
    sources = [
        f"{c.get('document_name', 'Unknown Document')} (Page {c.get('page_number', 'Unknown')})"
        for c in context[:TOP_K]
    ]
    
    primary_source = "\n             ".join(sources) if sources else "Unknown Source"
    
    result = chain.invoke({
        "history": request.history,
        "context": combined_context,
        "question": request.question
    })
    
    new_history = request.history + f"User: {request.question}\nBot: {result}\n"
    
    return {
        "response": result,
        "source": primary_source,
        "sources": sources,
        "history": new_history
    }


"""
uvicorn apps.web.server:app --reload
"""