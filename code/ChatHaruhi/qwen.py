import torch
from .BaseLLM import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
tokenizer_qwen = None
model_qwen = None
# Load model directly
def initialize_qwen():
    global model_qwen, tokenizer_qwen

    if model_qwen is None:
        model_qwen = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-7B-Chat",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    if tokenizer_qwen is None:
        tokenizer_qwen = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-7B-Chat", 
            use_fast=True,
            trust_remote_code=True
        )
            



    return model_qwen, tokenizer_qwen

def LLaMA_tokenizer(text):
    return len(tokenizer_qwen.encode(text))

class ChatQwen(BaseLLM):
    def __init__(self, model="qwen7b"):
        super(ChatQwen, self).__init__()
        self.model, self.tokenizer = initialize_qwen()
        self.messages = ""

    def initialize_message(self):
        self.messages = ""

    def ai_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def system_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def user_message(self, payload):
        self.messages = self.messages + "\n " + payload 

    def get_response(self):
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, self.messages, history=[])
            # print(response)
        return response
        
    def print_prompt(self):
        print(type(self.messages))
        print(self.messages)
