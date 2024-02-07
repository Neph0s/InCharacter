from .BaseLLM import BaseLLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes, flash_attn
tokenizer_LLaMA = None
model_LLaMA = None

def initialize_Mistral():
    global model_LLaMA, tokenizer_LLaMA

    if model_LLaMA is None:
        model_LLaMA = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    if tokenizer_LLaMA is None:
        tokenizer_LLaMA = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", trust_remote_code=True)

    return model_LLaMA, tokenizer_LLaMA

def LLaMA_tokenizer(text):
    return len(tokenizer_LLaMA.encode(text))

class ChatMistral(BaseLLM):
    def __init__(self, model="Mistral"):
        super(ChatMistral, self).__init__()
        self.model, self.tokenizer = initialize_Mistral()
        self.messages = ""

    def initialize_message(self):
        self.messages = "[INST]"

    def ai_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def system_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def user_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def get_response(self):
        with torch.no_grad():
            encodeds = self.tokenizer.encode(self.messages+"[/INST]", return_tensors="pt")
            generated_ids = self.model.generate(encodeds, max_new_tokens=2000, do_sample=True)
            decoded = self.tokenizer.batch_decode(generated_ids)

        return decoded[0].split("[/INST]")[1]
        
    def print_prompt(self):
        print(self.messages)
