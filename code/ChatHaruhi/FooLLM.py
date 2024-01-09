
from .BaseLLM import BaseLLM

# fooLLM do nothing but record the messages

class FooLLM(BaseLLM):

    def __init__(self ):
        self.messages = []

    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role":"AI","content":payload})

    def system_message(self, payload):
        self.messages.append({"role":"System","content":payload})

    def user_message(self, payload):
        self.messages.append({"role":"User","content":payload})

    def get_response(self):
        for message in self.messages:
            print(message["role"], ":", message["content"])
        response = input("Please input your response: ")
        return response
    
    def print_prompt(self):
        for message in self.messages:
            print(message["role"], ":", message["content"])
