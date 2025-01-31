from LLMS.Llms_doc import  get_answer_from_prompt_gemini, get_prompt_from_query, get_similar_embeddings, loaderdoc, split_text, vector_store
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
# from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import os

class PostMessageToPiyiView(APIView):
    def post(self, request):
        file_path = request.data.get("file_path")  # Ruta al archivo PDF
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        # text = serializer.data["text"]
        query = serializer.data["query"] # La pregunta del usuario

        try:
            
            prompt = get_prompt_from_query(query)
            response = get_answer_from_prompt_gemini(prompt,query)

            
            
            
            # Cargar y dividir el documento
            docs = loaderdoc(file_path)
            all_splits = split_text(docs)

            # Crear el vector store con los embeddings del documento
            vectorstore = vector_store(all_splits)

            # Obtener los documentos más similares a la pregunta
            similar_docs = get_similar_embeddings(vectorstore, query)
            
            # Generar una respuesta basada en los documentos similares
            response_text = " ".join([doc.page_content for doc in similar_docs])
            # print(response_text)
           
            # Estructurar la respuesta
            response_data = {
                "response": response_text,
                "relevant_documents": [
                    {"page_content": doc.page_content, "score": 1.0}  # Puedes calcular un score real si es necesario
                    for doc in similar_docs
                ],
                "status": "success",
            }

            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": str(e), "status": "error"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )