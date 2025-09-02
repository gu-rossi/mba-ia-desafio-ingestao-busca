import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PDF_PATH = os.getenv("PDF_PATH")
RAG_DIR = Path(__file__).parent.parent / PDF_PATH

def get_embedding_model():
    """
    Retorna o modelo de embedding baseado nas variáveis de ambiente disponíveis.
    Prioridade: Azure OpenAI > OpenAI
    """
    if os.getenv("AZURE_OPENAI_ENDPOINT"):
        print("Usando Azure OpenAI...")
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("A chave da API do OpenAI não está definida.")
        return AzureOpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    elif os.getenv("OPENAI_API_KEY"):
        print("Usando OpenAI...")
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

def get_pdf_files(directory: Path) -> list[Path]:
    return list(directory.glob("*.pdf"))

def ingest_pdf(pdf: Path):
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, add_start_index=False)
    chunks = text_splitter.split_documents(docs)

    enriched = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in chunks
    ]

    ids = [f"doc-{i}" for i in range(len(enriched))]

    store.add_documents(documents=enriched, ids=ids)

def main():
    pdf_files = get_pdf_files(RAG_DIR)
    for pdf in pdf_files:
        ingest_pdf(pdf)

if __name__ == "__main__":
    main()