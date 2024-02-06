import torch
from .BaseLLM import BaseLLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb
tokenizer_qwen = None
model_qwen = None
# Load model directly
def initialize_Qwen2LORA():
    global model_qwen, tokenizer_qwen

    if model_qwen is None:
        model_qwen = AutoModelForCausalLM.from_pretrained(
            "silk-road/ChatHaruhi_RolePlaying_qwen_7b",
            device_map="auto",
            trust_remote_code=True
        )
        model_qwen = model_qwen.eval()
        # model_qwen = PeftModel.from_pretrained(
        #     model_qwen,
        #     "silk-road/Chat-Haruhi-Fusion_B"
        # )

    if tokenizer_qwen is None:
        tokenizer_qwen = AutoTokenizer.from_pretrained(
            "silk-road/ChatHaruhi_RolePlaying_qwen_7b", 
            # use_fast=True,
            trust_remote_code=True
        )

    return model_qwen, tokenizer_qwen


def LLaMA_tokenizer(text):
    return len(tokenizer_qwen.encode(text))

class Qwen118k2GPT(BaseLLM):
    def __init__(self, model="qwen-118k"):
        super(Qwen118k2GPT, self).__init__()
        self.model, self.tokenizer = initialize_Qwen2LORA()
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
