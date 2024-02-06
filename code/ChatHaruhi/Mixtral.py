from .BaseLLM import BaseLLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, MixtralForCausalLM
import bitsandbytes, flash_attn
tokenizer_LLaMA = None
model_LLaMA = None

def initialize_Mixtral():
    global model_LLaMA, tokenizer_LLaMA

    if model_LLaMA is None:
        model_LLaMA = MixtralForCausalLM.from_pretrained(
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    if tokenizer_LLaMA is None:
        tokenizer_LLaMA = LlamaTokenizer.from_pretrained('NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO', trust_remote_code=True)

    return model_LLaMA, tokenizer_LLaMA

def LLaMA_tokenizer(text):
    return len(tokenizer_LLaMA.encode(text))

class ChatMixtral(BaseLLM):
    def __init__(self, model="Mixtral"):
        super(ChatMixtral, self).__init__()
        self.model, self.tokenizer = initialize_Mixtral()
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
            input_ids = self.tokenizer(self.messages, return_tensors="pt").input_ids.to("cuda")
            generated_ids = self.model.generate(input_ids, max_new_tokens=750, temperature=0.8, repetition_penalty=1.1, do_sample=True, eos_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True)
        return response
        
    def print_prompt(self):
        print(self.messages)
