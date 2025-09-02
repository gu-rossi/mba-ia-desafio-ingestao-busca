from search import search_prompt
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def get_model():
    """
    Retorna o modelo de chat baseado nas variáveis de ambiente disponíveis.
    Prioridade: Azure OpenAI > OpenAI
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Usando Azure OpenAI...")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("A chave da API do OpenAI não está definida.")
        return AzureChatOpenAI(azure_deployment=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"))
    elif os.getenv("OPENAI_API_KEY"):
        print("Usando OpenAI...")
        return ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-5-nano"), temperature=0.0)
    else:
        raise ValueError("Nenhuma API key válida encontrada. Configure AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY ou GOOGLE_API_KEY.")

chat_model = get_model()

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