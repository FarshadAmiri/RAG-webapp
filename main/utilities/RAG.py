from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from llama_index import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, download_loader
from llama_index import SimpleDirectoryReader
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores import MilvusVectorStore
import chromadb
from pathlib import Path
import accelerate
import torch
import time
from pprint import pprint


def load_model(model_name="TheBloke/Llama-2-7b-Chat-GPTQ", device='gpu'):
    
    # setting device
    if device == 'gpu':
        gpu=0
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        torch.cuda.get_device_name(0)
    elif device == 'cpu':
        device = torch.device('cpu')
        torch.cuda.set_device(device)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name
        # ,cache_dir='./model/'
        # ,use_auth_token=auth_token
        ,device_map='cuda'                 
        )

    # Define model
    model = AutoModelForCausalLM.from_pretrained(model_name
        ,cache_dir=r"C:\Users\user2\.cache\huggingface\hub"
        # ,cache_dir='./model/'
        # ,use_auth_token=auth_token
        ,device_map='cuda'  
        # , torch_dtype=torch.float16
        # ,low_cpu_mem_usage=True
        # ,rope_scaling={"type": "dynamic", "factor": 2}
        # ,load_in_8bit=True,
        ).to(device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    model_obj = {"model": model, "tokenizer": tokenizer, "streamer": streamer, "device": device,  }
    return model_obj


def llm_inference(plain_text, model, tokenizer, device, streamer=None, max_length=4000, ):
    input_ids = tokenizer(
        plain_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        )['input_ids'].to(device)
    
    output_ids = model.generate(input_ids
                        ,streamer=streamer
                        ,use_cache=True
                        ,max_new_tokens=float('inf')
                       )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return answer


def create_rag(path):
    db = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_or_create_collection("default")
    return


def add_docs(vdb_path: str, docs_paths: list):
    db = chromadb.PersistentClient(path = vdb_path)
    chroma_collection = db.get_or_create_collection("default")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)
    PyMuPDFReader = download_loader("PyMuPDFReader")
    loader = PyMuPDFReader()
    # Load documents
    for doc_path in docs_paths:
        document = loader.load(file_path=Path(doc_path), metadata=False)
        # Create indexes
        for doc in document:
            index.insert(doc)