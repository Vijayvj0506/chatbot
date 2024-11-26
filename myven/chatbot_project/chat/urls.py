from django.urls import path
from .views import chat_view,index

urlpatterns = [
    path('', index, name='index'), 
    path('api/chat/', chat_view, name='chatbot'),
]