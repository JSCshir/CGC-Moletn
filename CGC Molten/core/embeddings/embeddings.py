import torch
import numpy as np
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding
from typing import Tuple
from config.config import DEVICE


def compute_embeddings_for_chunks(pages_and_chunks: list[dict], embedding_model: SentenceTransformer) -> Tuple[list, torch.Tensor]:
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks]
    embeddings_tensor = embedding_model.encode(
        text_chunks,
        device=DEVICE,
        batch_size=16,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    for i, embedding in enumerate(embeddings_tensor):
        pages_and_chunks[i]["embedding"] = embedding.tolist()
    return pages_and_chunks, embeddings_tensor



def save_chunks_and_embeddings(pages_and_chunks: list[dict], file_path: str) -> None:
    df = pd.DataFrame(pages_and_chunks)
    df.to_csv(file_path, index=False)
    print(f"Saved chunks and embeddings to {file_path}")

def load_chunks_and_embeddings(file_path: str, device: str = DEVICE) -> Tuple[list, torch.Tensor]:
    df = pd.read_csv(file_path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    pages_and_chunks = df.to_dict(orient="records")
    embeddings_tensor = torch.tensor(np.array(df["embedding"].tolist()), dtype=torch.float32).to(device)
    print(f"Loaded chunks and embeddings from {file_path}")
    return pages_and_chunks, embeddings_tensor

