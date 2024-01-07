from django.shortcuts import render
from main.utilities.RAG import load_model, llm_inference

model_obj = load_model()
model = model_obj["model"]
tokenizer = model_obj["tokenizer"]
device = model_obj["device"]
streamer = model_obj["streamer"]


# Create your views here.
def chat_view(request):
    if request.method == "GET":
        return render(request, 'main/chat.html')