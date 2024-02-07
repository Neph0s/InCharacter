import torch
from .BaseLLM import BaseLLM
from transformers import AutoTokenizer, PhiForCausalLM
tokenizer_phi = None
model_phi = None
# Load model directly
def initialize_phi():
    global model_phi, tokenizer_phi

    if model_phi is None:
        model_phi = PhiForCausalLM.from_pretrained(
            "cognitivecomputations/dolphin-2_6-phi-2",
            local_files_only=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    if tokenizer_phi is None:
        tokenizer_phi = AutoTokenizer.from_pretrained(
            "cognitivecomputations/dolphin-2_6-phi-2", 
            local_files_only=True,
            use_fast=True,
        )
            



    return model_phi, tokenizer_phi

def LLaMA_tokenizer(text):
    return len(tokenizer_phi.encode(text))

class Chatphi(BaseLLM):
    def __init__(self, model="phi"):
        super(Chatphi, self).__init__()
        self.model, self.tokenizer = initialize_phi()
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
            # Prepare the model input with attention mask
            inputs = self.tokenizer(self.messages, return_tensors="pt", padding=True, truncation=True)
            attention_mask = inputs['attention_mask']
            
            # Generate the model output using the prepared input and attention mask
            outputs = self.model.generate(input_ids=inputs['input_ids'], attention_mask=attention_mask, max_length=114514)
            response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
        return response

        
    def print_prompt(self):
        print(type(self.messages))
        print(self.messages)
