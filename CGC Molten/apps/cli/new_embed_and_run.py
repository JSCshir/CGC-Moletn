

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from core.pdf.pdf_processing import load_all_pdfs_from_folder, initialize_spacy, process_pages_and_chunks
from core.embeddings.embeddings import compute_embeddings_for_chunks, save_chunks_and_embeddings, load_chunks_and_embeddings
from core.chat.chat_chain import create_chat_chain, handle_conversation
from sentence_transformers import SentenceTransformer
from config.config import PDF_FOLDER, SENTENCE_CHUNK_SIZE, DEVICE, MIN_TOKEN_LENGTH, EMBEDDINGS_FILE, MODEL_NAME



def main():
    pdf_folder = os.path.join(os.path.dirname(__file__), PDF_FOLDER)
    print("Using pdf_folder:", pdf_folder)
    
    pages_and_texts = load_all_pdfs_from_folder(pdf_folder)
    nlp = initialize_spacy()
    pages_and_chunks = process_pages_and_chunks(pages_and_texts, nlp, chunk_size=SENTENCE_CHUNK_SIZE)
    pages_and_chunks = [chunk for chunk in pages_and_chunks if chunk["chunk_token_count"] > MIN_TOKEN_LENGTH]
    
    embedding_model = SentenceTransformer(model_name_or_path=MODEL_NAME, device=DEVICE)

    pages_and_chunks, embeddings = compute_embeddings_for_chunks(pages_and_chunks, embedding_model)
    print("Embeddings computed on device:", embeddings.device)
    
    save_chunks_and_embeddings(pages_and_chunks, EMBEDDINGS_FILE)
    pages_and_chunks, embeddings = load_chunks_and_embeddings(EMBEDDINGS_FILE, device=DEVICE)
    
    chain = create_chat_chain()
    handle_conversation(chain, embedding_model, pages_and_chunks, embeddings)

if __name__ == "__main__":
    main()



