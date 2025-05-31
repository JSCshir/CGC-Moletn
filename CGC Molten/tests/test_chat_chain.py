# run_batch_tests.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import os
import sys
from core.embeddings.embeddings import load_chunks_and_embeddings
from core.chat.chat_chain import create_chat_chain
from core.retrieval.retrieval import search_and_retrieve
from config.config import MODEL_NAME
from sentence_transformers import SentenceTransformer

INPUT_CSV = "/home/jacobhardy/CGC Molten/tests/Embedding_Test_Questions.csv"
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "molten_epme_results2.csv")
EMBEDDINGS_FILE = "text_chunks_and_embeddings_df.csv"
DEVICE = "cuda"

def load_input_questions(path):
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_output_csv(path, rows):
    with open(path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Question",
            "EPME Answer",
            "EPME Sources",
            "Molten Answer",
            "Molten Sources"
        ])
        writer.writerows(rows)

def main():
    print("🧠 Loading embedding model and data...")
    pages_and_chunks, embeddings = load_chunks_and_embeddings(EMBEDDINGS_FILE, device=DEVICE)
    embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embedding_model.to(DEVICE)
    chain = create_chat_chain()

    input_rows = load_input_questions(INPUT_CSV)
    output_rows = []

    print(f"📄 Running batch test on {len(input_rows)} questions...\n")

    for row in input_rows:
        question = row["Question"]
        epme_answer = row["Answer"]
        epme_ref = row["Reference"]

        context = search_and_retrieve(question, embedding_model, pages_and_chunks, embeddings, top_k=5)
        if not context:
            molten_answer = "Sorry, I couldn't find relevant information."
            molten_sources = "N/A"
        else:
            combined_context = "\n\n".join([c["text"] for c in context])
            sources_list = [
                f"{c.get('document_name', 'Unknown')} (Page {c.get('page_number', '?')})"
                for c in context
            ]
            molten_sources = "; ".join(sources_list)

            molten_answer = chain.invoke({
                "history": "",
                "context": combined_context,
                "question": question
            })

        output_rows.append([
            question,
            epme_answer,
            epme_ref,
            molten_answer,
            molten_sources
        ])

    write_output_csv(OUTPUT_CSV, output_rows)
    print(f"\n✅ Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    # Add project root to sys.path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
