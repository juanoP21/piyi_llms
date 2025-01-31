from LLMS.Llms_doc import get_embedding
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import os

class PostMessageToPiyiView(APIView):
     def post(self, request):
        texts = request.data.get("texts")  # Obtener el valor de "texts"

        # Validar que 'texts' existe y es una lista
        if not texts or not isinstance(texts, list):
            return Response({"error": "Invalid input. Expected a list of texts."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            embeddings = get_embedding(texts)  # Pasar la lista de textos
            return Response({"embeddings": embeddings}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)