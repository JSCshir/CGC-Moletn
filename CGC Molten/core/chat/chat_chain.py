import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from config.config import CHAT_TEMPLATE, LLAMA_MODEL, TEMPERATURE, TOP_K
from core.pdf.pdf_processing import apply_acronym_dict
from core.retrieval.retrieval import search_and_retrieve


def create_chat_chain() -> any:
    prompt = ChatPromptTemplate.from_template(CHAT_TEMPLATE)
    model = OllamaLLM(model=LLAMA_MODEL, temperature=TEMPERATURE)
    chain = prompt | model
    return chain

def handle_conversation(chain, embedding_model, pages_and_chunks, embeddings) -> None:
    conversation_history = ""  
    print("Welcome to the AI chatbot. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        user_input = apply_acronym_dict(user_input)

        context = search_and_retrieve(user_input, embedding_model, pages_and_chunks, embeddings, top_k=TOP_K)
        if not context:
            print("Bot: Sorry, I couldn't find relevant information.")
            continue
        
        conversation_history += f"User: {user_input}\n"
        
        aggregated_context = "\n\n".join([c["text"] for c in context])
        
        result = chain.invoke({
            "history": conversation_history,
            "context": aggregated_context,
            "question": user_input
        })
        
        conversation_history += f"Bot: {result}\n"
        
        print("Bot:", result)
        print()
        
        for c in context:
            source_info = f"Source: {c['document_name']} (Page {c['page_number']})"
            print(source_info)
