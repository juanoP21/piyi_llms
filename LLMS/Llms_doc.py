from __future__ import annotations
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_core.prompts import  ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))


class GoogleVertexAIEmbeddings(Embeddings):
    def __init__(self, model_name="text-embedding-005", dimensionality=256):
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.dimensionality = dimensionality

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        task = "RETRIEVAL_DOCUMENT"
        inputs = [TextEmbeddingInput(text, task) for text in texts]
        kwargs = dict(output_dimensionality=self.dimensionality) if self.dimensionality else {}
        embeddings = self.model.get_embeddings(inputs, **kwargs)
        # Extraer los valores del embedding
        return [embedding.values for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        # Extraer los valores del embedding para una sola consulta
        return self.embed_documents([text])[0]

def loaderdoc(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def vector_store(all_splits):
    embeddings_model = GoogleVertexAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings_model)
    return vectorstore

def get_similar_embeddings(vectorstore, query):
    embeddings_model = GoogleVertexAIEmbeddings()
    query_embedding = embeddings_model.embed_query(query)
    similar_docs = vectorstore.similarity_search_by_vector(query_embedding, k=5)
    return similar_docs

def get_prompt_from_query(query,embeddings):
    #prompt = "You are a virtual assistant. Use the following context to answer the question: \n"

    print("Selected embeddings: ")
    
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", "Eres un asistente virtual. Usa el siguiente contexto para contestar la pregunta: \n"),
            ("user", "{query}",("context",embeddings) )
        ])
    return prompt_template

def get_answer_from_prompt_gemini(prompt,query):
    while True:
        try:
            chain = prompt | llm
            response = chain.invoke({"user_input": query})

            return str(response)
        except Exception as e:
            print("GPT Error")
            print(e)
            

