# Desafio MBA Engenharia de Software com IA - Full Cycle

Este projeto implementa uma solução de Retrieval-Augmented Generation (RAG) que permite fazer perguntas sobre um documento PDF. A solução utiliza o LangChain para orquestrar o processo, um banco de dados PostgreSQL com a extensão `pgvector` para armazenar os embeddings do documento, e modelos de linguagem da OpenAI, Google ou Azure para gerar as respostas.

## Arquitetura da Solução

A solução é composta por três scripts principais:

1.  **`src/ingest.py`**: Responsável por carregar o documento PDF, dividi-lo em chunks, gerar os embeddings e armazená-los no banco de dados PGVector.
2.  **`src/search.py`**: Realiza a busca por similaridade no banco de dados com base na pergunta do usuário e monta o prompt final com o contexto recuperado.
3.  **`src/chat.py`**: Orquestra a interação com o usuário, enviando o prompt para o modelo de linguagem e exibindo a resposta.

## Como Executar a Solução

Siga os passos abaixo para configurar e executar o projeto.

### 1. Pré-requisitos

- Python 3.10+
- Docker e Docker Compose

### 2. Configuração do Ambiente

Primeiro, configure o ambiente e instale as dependências.

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd <NOME_DO_DIRETORIO>
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure as variáveis de ambiente:**
    - Renomeie o arquivo `.env.example` para `.env`.
    - Abra o `.env` e preencha com suas chaves de API (OpenAI, Google ou Azure).
    - **Importante**: Caso os seus arquivos PDF estejam na raiz do projeto, não será necessário atualizar a variável `PDF_PATH` no arquivo `.env` pois o caminho padrão já está configurado com a variável vazia, se optar por incluir em uma subpasta chamada `docs` dentro da raiz do projeto, atualize a variável `PDF_PATH` com o caminho relativo. Por exemplo:
      ```
      PDF_PATH=docs
      ```

5.  **Inicie o banco de dados com Docker:**
    ```bash
    docker-compose up -d
    ```
    Este comando irá iniciar um container PostgreSQL com a extensão `pgvector` habilitada.

### 3. Ingestão dos Dados

Com o ambiente configurado, execute o script de ingestão para processar o PDF e popular o banco de dados.

```bash
python3 src/ingest.py
```

O script irá ler os PDFs especificados dentro da pasta em `PDF_PATH`, gerar os embeddings e armazená-los na coleção `rag_collection` do banco de dados.

### 4. Inicie o Chat

Após a ingestão dos dados, você pode iniciar o chat para fazer perguntas sobre o seu documento.

```bash
python3 src/chat.py
```

O sistema irá carregar o modelo de linguagem configurado e você poderá fazer perguntas no terminal. Para sair, digite `/quit`.

**Exemplo de interação:**
```
$ python3 src/chat.py
Usando OpenAI...
Chat iniciado! Digite '/quit' para sair.

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?

RESPOSTA: R$ 10.000.000,00.
---------
```

### Criando uma API Key

Para executar os códigos, é necessário ter chaves de API da OpenAI ou do Google. Siga as instruções abaixo para obtê-las.

#### OpenAI

1.  Acesse [platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys).
2.  Faça login e clique em "Create new secret key".
3.  Copie a chave e cole no arquivo `.env` na variável `OPENAI_API_KEY`.

#### Google Gemini

1.  Acesse o [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key?hl=pt-BR).
2.  Faça login e clique em "Create API Key".
3.  Copie a chave e cole no arquivo `.env` na variável `GOOGLE_API_KEY`.

## Ordem de Precedência dos Provedores

A aplicação foi projetada para ser flexível, permitindo o uso de diferentes provedores de modelos de linguagem (LLMs). A seleção do provedor é determinada pela presença de variáveis de ambiente específicas no seu arquivo `.env`, seguindo esta ordem de prioridade:

1.  **Azure OpenAI**: Se a variável `AZURE_OPENAI_ENDPOINT` estiver definida, a aplicação usará os serviços do Azure OpenAI.
2.  **OpenAI**: Se a variável do Azure não estiver configurada, mas a `OPENAI_API_KEY` estiver presente, a aplicação usará o modelo da OpenAI.
3.  **Google Gemini**: Caso nenhuma das opções anteriores esteja configurada, a aplicação procurará pela `GOOGLE_API_KEY` e usará o modelo Gemini do Google.

Se nenhuma dessas variáveis de ambiente for encontrada, o sistema lançará um erro, informando que uma chave de API válida é necessária.
