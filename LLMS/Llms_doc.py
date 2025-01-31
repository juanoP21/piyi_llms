from __future__ import annotations

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_community.document_loaders import PyPDFLoader

def get_embedding(texts: list[str]):
    
    dimensionality = 256
    # The task type for embedding. Check the available tasks in the model's documentation.
    task = "RETRIEVAL_DOCUMENT"
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    inputs = [TextEmbeddingInput(text, task) for text in texts]
    kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}
    embeddings = model.get_embeddings(inputs, **kwargs)

    print(embeddings)
    return [embedding.values for embedding in embeddings]


def loaderdoc(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    # vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return 