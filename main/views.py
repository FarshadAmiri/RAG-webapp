from django.shortcuts import render, redirect
from main.utilities.RAG import load_model, llm_inference
from main.models import Vector_db, Document
import os

documents_db_path = "documents_db"

model_obj = load_model()
model = model_obj["model"]
tokenizer = model_obj["tokenizer"]
device = model_obj["device"]
streamer = model_obj["streamer"]


# Create your views here.
def chat_view(request, chat_id=None):
    user = request.user
    if request.method == "GET":
        chat_threads = Vector_db.objects.filter(user=user).values()
        context = {"chat_threads": chat_threads}
        return render(request, 'main/chat.html', context)
    

# Create your views here.
def create_rag_view(request,):
    user = request.user
    if request.method == "POST":
        uploaded_files = request.FILES.getlist('files')
        print(f"uploaded_files: {uploaded_files}")
        print(f"request.FILES: {request.FILES}")
        print(f"request: {request}")
        for file in uploaded_files:
            file_name = file.name
            file_size = file.size
            print(f"file_name: {file_name}")
            print(f"file_size: {file_size}")
            print(f"file: {type(file)}")
            folder_path = os.path.join(documents_db_path, user.username)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'wb') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
        return redirect('main:main_chat')