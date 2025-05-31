# apps/cli/cli_chat.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from core.embeddings.embeddings import load_chunks_and_embeddings
from core.chat.chat_chain import create_chat_chain
from core.retrieval.retrieval import search_and_retrieve
from sentence_transformers import SentenceTransformer
from datetime import datetime

LOG_FILE = os.path.join(os.path.dirname(__file__), "molten_cli_log.txt")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log_interaction(question, response, sources):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"---\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Answer: {response}\n")
        f.write(f"Sources:\n")
        for s in sources:
            f.write(f"  - {s}\n")
        f.write(f"\n")


# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def main():
    embeddings_file = os.path.join(os.path.dirname(__file__), "../../text_chunks_and_embeddings_df.csv")
    device = "cuda"

    print("🔌 Loading precomputed embeddings...")
    pages_and_chunks, embeddings = load_chunks_and_embeddings(embeddings_file, device)
    
    print("🧠 Loading embedding model...")
    embedding_model = SentenceTransformer("all-mpnet-base-v2", device=device)
    embedding_model.to(device)

    print("🪄 Spinning up CGC Molten chat chain...")
    chain = create_chat_chain()

    print("\n🧭 CGC Molten Terminal Chat - Type 'exit' to quit\n")
    history = ""

    while True:
        question = input("You: ")
        if question.strip().lower() == "exit":
            break

        context = search_and_retrieve(question, embedding_model, pages_and_chunks, embeddings, top_k=5)

        if not context:
            print("Molten: Sorry, I couldn't find anything useful.\n")
            continue

        combined_context = "\n\n".join([c["text"] for c in context])

        result = chain.invoke({
            "history": history,
            "context": combined_context,
            "question": question
        })

        history += f"User: {question}\nBot: {result}\n"

        print(f"\nMolten: {result}")
        print("Sources:")
        for c in context:
            print(f" - {c['document_name']} (Page {c['page_number']})")
        print()

        sources_list = [
            f"{c.get('document_name', 'Unknown')} (Page {c.get('page_number', '?')})"
            for c in context
            ]

        log_interaction(question, result, sources_list)


if __name__ == "__main__":
    main()
