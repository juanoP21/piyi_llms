# from .views import PostMessageToPiyiView
from django.urls import path
from LLMS.views import PostMessageToPiyiView

urlpatterns = [
    path('embed/', PostMessageToPiyiView.as_view(), name='embed_text'),  # Esta es la ruta correcta

]