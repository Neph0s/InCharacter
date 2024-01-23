from transformers import AutoTokenizer, AutoModelForCausalLM
class ChracterLLM:
    def __init__(self, characterName):
        self.name=characterName
        self.ModelName=f"fnlp/character-llm-{characterName}-7b-wdiff"
    def ask(self,prompt):
       # Load model directly
        meta_prompt = """I want you to act like {character}. I want you to respond and answer like {character}, using the tone, manner and vocabulary {character} would use. You must know all of the knowledge of {character}. 

        The status of you is as follows:
        Location: {loc_time}
        Status: {status}

        The interactions are as follows:"""
        character=self.name
        loc_time = "Coffee Shop - Afternoon"
        status = f'{character} is casually chatting with a man from the 21st century.'
        prompt = prompt+'\n\n'
        tokenizer = AutoTokenizer.from_pretrained(self.ModelName)
        model = AutoModelForCausalLM.from_pretrained(self.ModelName)
        inputs = tokenizer([prompt], return_tensors="pt")
        outputs = model.generate(**inputs, do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=300)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)



bot=ChracterLLM("cleopatra")

print(bot.ask("who are you?"))