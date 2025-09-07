from search import search_prompt
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

chat_model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"), temperature=0.0)

def main():
    try:
        chain = search_prompt | chat_model
        print("Chat iniciado! Digite '/quit' para sair.")
    except Exception as e:
        print(f"Erro ao inicializar o chat: {e}")
        return
    
    while True:
        query = input("\nPERGUNTA: ")
        
        if query.strip().lower() == "/quit":
            print("Chat encerrado. Até logo!")
            break
        
        if query.strip():
            try:
                result = chain.invoke({"query": query})
                print(f"\nRESPOSTA: {result.content}\n---------")
            except Exception as e:
                print(f"Erro ao processar a pergunta: {e}")
        else:
            print("Por favor, digite uma pergunta válida ou '/quit' para sair.")

if __name__ == "__main__":
    main()
