from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import accelerate
import torch
import time
from pprint import pprint


# setting device
gpu=0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(device)
torch.cuda.get_device_name(0)


# Define model name and hf token
name = "TheBloke/Llama-2-7b-Chat-GPTQ"
# name = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"

# hugginf face auth token
file_path = "huggingface_credentials.txt"
with open(file_path, "r") as file:
    auth_token = file.read().strip()



# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(name
    # ,cache_dir='./model/'
    ,use_auth_token=auth_token
    ,device_map='cuda'                 
    )


# Define model
model = AutoModelForCausalLM.from_pretrained(name
    ,cache_dir=r"C:\Users\user2\.cache\huggingface\hub"
    # ,cache_dir='./model/'
    ,use_auth_token=auth_token
    ,device_map='cuda'
    # , torch_dtype=torch.float16
    # ,low_cpu_mem_usage=True
    # ,rope_scaling={"type": "dynamic", "factor": 2}
    # ,load_in_8bit=True,
    ).to(device)


streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)