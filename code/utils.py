import pdb 
import os
import re 
import random 
import openai
import json
import logging
import time  
import jsonlines 
import requests 
import io
import pickle

logger = logging.getLogger(__name__)

file_handler = logging.FileHandler('log.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)  # 设置日志级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
console_handler.setFormatter(formatter)
 

cache_sign = True

with open('config.json', 'r') as f:
	config = json.load(f)

#openai.proxy = config['proxy']
if config.get('proxy', None):
	openai.proxy = config['proxy']

if config.get('openai_apibase', None):
	openai.api_base = config['openai_apibase']

cache = None 
def cached(func):
	def wrapper(*args, **kwargs):		
		global cache
		cache_path = 'cache.pkl'
		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				cache = pickle.load(open(cache_path, 'rb'))  

		key = ( func.__name__, str(args), str(kwargs.items()))
		
		if (cache_sign and key in cache and cache[key] not in [None, '[TOKEN LIMIT]']) :
			return cache[key]
		else:
			result = func(*args, **kwargs)
			if result != 'busy' and result != None:
				cache[key] = result
				pickle.dump(cache, open(cache_path, 'wb'))
			return result

	return wrapper

def get_response(sys_prompt, inputs, model='gpt4', nth_generation=0):
	model = model.lower().replace(' ', '')
	if model.startswith('gpt-3.5'):
		model = 'gpt-3.5-turbo-1106'
	elif model.startswith('gpt-4'):
		model = 'gpt-4-1106-preview'
	
	return get_response_gpt(sys_prompt, inputs, model, nth_generation=nth_generation)

from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=config['openai_apikey'],
)

@cached 
def get_response_gpt(sys_prompt, inputs, model='gpt-4', retry_count=0, nth_generation=0):

	query = [ {'role': 'system', 'content': sys_prompt}]
	if len(inputs) > 0:
		query.append({'role': 'user', 'content': inputs})
	
	try:
		temperature = 0.2 if nth_generation else 0 
		#logger.info('ChatGPT SysPrompt:  ' + sys_prompt[:100])
		#logger.info('ChatGPT Input:  ' + inputs[:100])
		response = client.chat.completions.create(
			model= model ,  # 对话模型的名称
			messages=query,
			temperature=temperature,  # 值在[0,1]之间，越大表示回复越具有不确定性
			top_p=1,
			frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
			presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容,
		)

		#logger.info('GPT Output: ' + response.choices[0].message.content[:100])
		return response.choices[0].message.content

	except openai.BadRequestError as e:
		logger.exception(e)
		
		return '[TOKEN LIMIT]'

	except Exception as e:
		# unknown exception
		logger.exception(e)

		if retry_count < 2:
			time.sleep(5)
			logger.warn("[OPEN_AI] RateLimit exceed, 第{}次重试".format(retry_count+1))
			return get_response_gpt(sys_prompt, inputs, model, retry_count+1) 

		print(f'Fail to get response after {retry_count} retry')

def string2json(llm_response):
	llm_response = llm_response.strip("`")
	if llm_response.startswith('json'):
		llm_response = llm_response[4:]
	
	try:
		json_response = json.loads(llm_response)
	except:
		try:
			llm_response = llm_response[llm_response.find("{"):]
			json_response = json.loads(llm_response)
		except:
			return False
	
	return json_response

def get_response_json(**kwargs):
	
	nth_generation = 0


	while (True):
		response = get_response(**kwargs, nth_generation=nth_generation)
		#print(f'{nth_generation} generation: {response[:100]}')

		json_response = string2json(response)
		#print(f'parse results: {json_response}')

		if json_response:
			break 
		else:
			nth_generation += 1

	return json_response

def avg(lst):
	return sum(lst)/len(lst)

def std(lst):
	import math
	avg_score = avg(lst)
	return math.sqrt(sum([(s-avg_score)**2 for s in lst])/ (len(lst) - 1))


if __name__ == '__main__':
	print(get_response('Act as a calculator', '123+456=?', 'gpt-3.5-turbo'))
		


