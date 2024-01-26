import torch
from transformers import AutoTokenizer, AutoModel,LlamaForCausalLM
model_LLaMA = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer_LLaMA = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", 
    use_fast=True
)

with torch.no_grad():
    # Prepare your input text (this could be your `self.messages`)
    input_text = ""
    # Tokenize the input text
    input_ids = tokenizer_LLaMA.encode(input_text, return_tensors='pt')

    # Generate the response
    output_ids = model_LLaMA.generate(input_ids,num_return_sequences=1)
    
    # Decode the generated ids to text
    response = tokenizer_LLaMA.decode(output_ids[0], skip_special_tokens=True)

    print(response)
