import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

load_dotenv()

def get_embedding_model():
    """
    Retorna o modelo de embedding baseado nas variáveis de ambiente disponíveis.
    Prioridade: Azure OpenAI > OpenAI
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("A chave da API do OpenAI não está definida.")
        return AzureOpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    elif os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    else:
        raise ValueError("Nenhuma API key válida encontrada. Configure AZURE_OPENAI_ENDPOINT, OPENAI_API_KEY ou GOOGLE_API_KEY.")

embeddings = get_embedding_model()

store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

@chain
def search_prompt(input_dict:dict) -> dict:
    query = input_dict["query"]
    results = store.similarity_search_with_score(query, k=10)
    context = "\n\n".join([doc.page_content for doc, _ in results])

    template = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE
    )

    prompt = template.format(contexto=context, pergunta=query)

    return prompt
