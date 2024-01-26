import torch
from .BaseLLM import BaseLLM
from transformers import AutoTokenizer, AutoModel,LlamaForCausalLM

tokenizer_LLaMA = None
model_LLaMA = None

def initialize_LLaMA():
    global model_LLaMA, tokenizer_LLaMA

    if model_LLaMA is None:
        model_LLaMA = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )

    if tokenizer_LLaMA is None:
        tokenizer_LLaMA = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf", 
            use_fast=True
        )

    return model_LLaMA, tokenizer_LLaMA

def LLaMA_tokenizer(text):
    return len(tokenizer_LLaMA.encode(text))

class ChatLLaMA(BaseLLM):
    def __init__(self, model="llama-2-7b"):
        super(ChatLLaMA, self).__init__()
        self.model, self.tokenizer = initialize_LLaMA()
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
            # Prepare your input text (this could be your `self.messages`)
            input_text = str(self.messages)
            # Tokenize the input text
            input_ids = tokenizer_LLaMA.encode(input_text, return_tensors='pt')

            # Generate the response
            output_ids = model_LLaMA.generate(input_ids,num_return_sequences=1)
            
            # Decode the generated ids to text
            response = tokenizer_LLaMA.decode(output_ids[0], skip_special_tokens=True)

            print(response)
        return response
        
    def print_prompt(self):
        print(type(self.messages))
        print(self.messages)
