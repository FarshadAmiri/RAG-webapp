from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
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
